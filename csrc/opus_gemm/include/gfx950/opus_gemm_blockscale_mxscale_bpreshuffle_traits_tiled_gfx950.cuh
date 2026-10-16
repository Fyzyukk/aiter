#ifndef OPUS_BP_TILED_TRAITS_BASE
#define OPUS_BP_TILED_TRAITS_BASE
// SPDX-License-Identifier: Apache-2.0

#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"

namespace opus_bpreshuffle_gfx950::tiled::schedule_0 {
// SPDX-License-Identifier: Apache-2.0

struct opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int VEC_SF = 4;
static constexpr int VEC_SF_PAIR = 2 * VEC_SF;
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 : opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int LOOP_UNROLL = 2;
static constexpr bool RESET_SFA_BEFORE_LOAD = false;
static constexpr int B_M=256;
static constexpr int B_N=256;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M / 2;
static constexpr int HALF_B_N = B_N / 2;
static_assert(NUM_WAVES == 4);
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(HALF_B_M % (W_M * T_M) == 0);
static_assert(HALF_B_N % (W_N * T_N) == 0);
static_assert(B_K % (W_K * T_K) == 0);
static constexpr int E_M = HALF_B_M / (W_M * T_M);
static constexpr int E_N = HALF_B_N / (W_N * T_N);
static constexpr int E_K = B_K / (W_K * T_K);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int OUTPUT_TILES_PER_WG = 1;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_SCALE_SF = 4;
static constexpr int VEC_SCALE_SF_PAIR = 8;
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MFMA_SCALE_GROUP_K = 32;
static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_sub = smem_linear_wave / B_K;
static constexpr int smem_m_rep = HALF_B_M / smem_sub;
static constexpr int smem_n_rep = HALF_B_N / smem_sub;
static constexpr int smem_padding = 32;
static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
static_assert(NUM_KGROUPS == 4);
static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int a_ds_read_insts = E_M * W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SCALE_PANEL_K_CAPACITY = 64;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M;
static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_PITCH = B_M;
static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;
static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
static_assert(E_M == 4 && E_N == 4 && E_K == 1);
static_assert(VEC_SCALE_SF == E_M);
static_assert(VEC_SCALE_SF_PAIR == 2 * VEC_SCALE_SF);
static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS == SCALE_PANEL_K_CAPACITY);
static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_scale_reset_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr bool RESET_SFA_BEFORE_LOAD = true;
};

__host__ __device__ inline int opus_mxscale_bpreshuffle_ceil_div(int a, int b) {
return (a + b - 1) / b;
}
template<int BlockM, int BlockN, int WaveM, int WaveN, int Prefetch,
int WaveK = 1, int Output = 0, int BCache = 0,
int FixedK = 0, bool PadN = false, bool ReuseBScale = false>
struct opus_gemm_small_register_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K = WaveK;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * WaveK * 64;
static constexpr int E_M = BlockM / (WaveM * W_M);
static constexpr int E_N = BlockN / (WaveN * W_N);
static constexpr int PREFETCH = Prefetch, WAVE_K = WaveK;
static constexpr int B_CACHE = BCache, OUTPUT = Output;
static constexpr int FIXED_K = FixedK;
static constexpr bool N_TAIL = PadN, REUSE_B_SCALE = ReuseBScale;
static_assert(BlockM % (WaveM * W_M) == 0 && BlockN % (WaveN * W_N) == 0);
static_assert(E_M > 0 && E_M <= 4 && E_N > 0 && Prefetch > 0);
static_assert(BLOCK_SIZE <= 512 && WaveK > 0);
static_assert(Output == 0 || Output == 3 || Output == 4);
static_assert(BCache >= 0 && BCache <= 3);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % 128 == 0));
static_assert(!PadN || Output == 4);
static_assert(!ReuseBScale || 128 % BlockN == 0);
static_assert(Output != 4 || WaveK > 1);
static_assert((WaveK - (Output == 4 ? 0 : 1)) * BlockM * BlockN * sizeof(float) <= 160 * 1024);
};
template<int BlockM, int BlockN, int WaveM, int WaveN,
int Stages, int Cluster = 1, int Output = 2,
bool RegisterScales = false, bool XorLds = false, bool EarlyScaleLoads = false,
bool PrefetchBeforeRead = false, bool ReadOnlyDrain = false,
int SplitK = 1, int ReduceVec = 4, int ReduceBlock = 128,
int StoreCache = 0, int FixedK = 0, bool FineMLoads = false>
struct opus_gemm_small_lds_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * 64;
static constexpr int WARP_SIZE = 64, NUM_WAVES = WaveM * WaveN;
static constexpr int E_M = BlockM / (WaveM * 16), E_N = BlockN / (WaveN * 16), E_K = 1;
static constexpr int HALF_B_M = BlockM, HALF_B_N = BlockN;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_SCALE_A = 16;
static constexpr int smem_linear_wave = 1024, smem_padding = XorLds ? 0 : 32;
static constexpr int smem_m_rep = BlockM / 8, smem_n_rep = BlockN / 8;
static constexpr int A_STAGE = BlockM / 8 * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = BlockN / 8 * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = Stages, CLUSTER = Cluster;
static constexpr int OUTPUT = Output;
static constexpr bool REGISTER_SCALES = RegisterScales, XOR_LDS = XorLds;
static constexpr bool EARLY_SCALE_LOADS = EarlyScaleLoads;
static constexpr bool PREFETCH_BEFORE_READ = PrefetchBeforeRead;
static constexpr bool READ_ONLY_DRAIN = ReadOnlyDrain;
static constexpr bool FINE_M_LOADS = FineMLoads;
static constexpr int SPLIT_K = SplitK, REDUCE_VEC = ReduceVec, REDUCE_BLOCK = ReduceBlock;
static constexpr int STORE_CACHE = StoreCache, FIXED_K = FixedK;
static constexpr int MAX_K = 16384, MAX_LOOPS = (MAX_K / B_K + SplitK - 1) / SplitK;
static constexpr int B_GROUPS = (BlockN + 127) / 128;
static constexpr int MAX_SFA_BYTES = BlockM * MAX_LOOPS;
static constexpr int MAX_LDS_BYTES = Stages * (A_STAGE + B_STAGE) +
(RegisterScales ? 0 : (BlockM + B_GROUPS) * MAX_LOOPS);
static constexpr int VMEM_TILE = (BlockM + BlockN) * 128 / (BLOCK_SIZE * 16) +
(RegisterScales ? E_M + B_GROUPS : 0);
static_assert(BlockM % (T_M * 16) == 0 && BlockN % (NUM_WAVES * 16) == 0);
static_assert(FineMLoads || BlockM % (BLOCK_SIZE / 8) == 0);
static_assert(E_M > 0 && E_M <= (FineMLoads ? 8 : 4) && E_N > 0 && MAX_LOOPS >= Stages);
static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4 || SplitK == 8);
static_assert(ReduceVec >= 4 && ReduceVec % 4 == 0 && ReduceVec <= 16);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= MAX_K && FixedK % 128 == 0));
static_assert(Cluster > 0 && Stages % Cluster == 0 && Stages >= 2 * Cluster);
static_assert(MAX_LDS_BYTES <= 160 * 1024 && B_GROUPS * MAX_LOOPS <= BLOCK_SIZE);
static_assert(Output == 1 || Output == 2);
static_assert(!XorLds || (NUM_WAVES == 4 && A_STAGE % 4096 == 0 && B_STAGE % 4096 == 0));
static_assert(Output != 1 || (BlockM * (BlockN + 8) * 2 <= A_STAGE + B_STAGE &&
BlockM * BlockN % (BLOCK_SIZE * 8) == 0));
static constexpr int lds_bytes(int k) {
const int loops = (k / B_K + SplitK - 1) / SplitK;
const int stages = loops < NUM_STAGES ? loops : NUM_STAGES;
return stages * (A_STAGE + B_STAGE) + (RegisterScales ? 0 : (BlockM + B_GROUPS) * loops);
}
};
using opus_gemm_mxscale_bpreshuffle_small_register_16x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 6, 1, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_16x16_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 16, 1, 1, 2, 8>;
using opus_gemm_mxscale_bpreshuffle_small_register_32x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<32, 32, 1, 1, 3, 4, 3, 0, 0, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 1, 4, 8, 2, 2, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_64x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 4, 1, 2, false, false, true, true, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_96x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<96, 64, 2, 2, 4, 1, 1, false, false, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_64x128_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<64, 128, 4, 2, 6, 2>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 2, 2, 4, 1, 2, true>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 2, 2, 4, 1, 2, true, true>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x128_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 128, 2, 2, 4, 2, 2, true, true, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x16_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 16, 1, 1, 3, 8, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 3, 4, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_16x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 2, 8, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<32, 32, 1, 1, 2, 8, 3, 3, 0, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x64_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<32, 64, 1, 1, 2, 4, 3, 3, 0, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_deep_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 1, 4, 12, 4, 2, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_deep_64x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 8, 2, 2, false, false, true>;
constexpr int opus_gemm_mxscale_bpreshuffle_small_gcd(int a, int b) { while (b) { int r=a%b; a=b; b=r; } return a; }
template<class Baseline, int Ahead>
struct opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits : Baseline {
static constexpr int B_AHEAD = Ahead, B_SLOTS = Ahead + 1;
static constexpr int RING_PERIOD = Baseline::NUM_STAGES * B_SLOTS / opus_gemm_mxscale_bpreshuffle_small_gcd(Baseline::NUM_STAGES, B_SLOTS);
static constexpr int B_STAGE = 0;
static constexpr int C_BYTES = Baseline::OUTPUT == 1 ? Baseline::B_M * (Baseline::B_N + 8) * 2 : 0;
static constexpr int MAX_LDS_BYTES = Baseline::NUM_STAGES * Baseline::A_STAGE +
(Baseline::REGISTER_SCALES ? 0 : (Baseline::B_M + Baseline::B_GROUPS) * Baseline::MAX_LOOPS);
static_assert(Ahead >= 1 && Ahead <= 3 && Baseline::T_K == 1);
static_assert(MAX_LDS_BYTES <= 160 * 1024 && C_BYTES <= 160 * 1024);
static constexpr int lds_bytes(int k) {
const int loops = (k / Baseline::B_K + Baseline::SPLIT_K - 1) / Baseline::SPLIT_K;
const int stages = loops < Baseline::NUM_STAGES ? loops : Baseline::NUM_STAGES;
const int matrix_scale = stages * Baseline::A_STAGE +
(Baseline::REGISTER_SCALES ? 0 : (Baseline::B_M + Baseline::B_GROUPS) * loops);
return matrix_scale > C_BYTES ? matrix_scale : C_BYTES;
}
};
template<int Actual> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits;
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9043> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9044> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_64x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9045> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_96x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9046> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_64x128_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9047> { using type = opus_gemm_mxscale_bpreshuffle_small_regscale_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9049> { using type = opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x128_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9055> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_deep_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9056> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_deep_64x64_traits_gfx950; };
template<int Actual, int Ahead> using opus_gemm_mxscale_bpreshuffle_small_direct_b_traits = opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits<typename opus_gemm_mxscale_bpreshuffle_small_baseline_traits<Actual>::type, Ahead>;
template<int BM, int BN, int WaveK>
struct opus_gemm_mxscale_bpreshuffle_register_split_traits
: opus_gemm_small_register_traits_gfx950<BM, BN, 1, 1, 3, WaveK, 3, 3> {
static constexpr int GLOBAL_SPLIT_K = 4;
static_assert(WaveK == 1 || WaveK == 2);
static_assert((BM == 16 && (BN == 16 || BN == 32)) ||
(BM == 32 && (BN == 32 || BN == 64)));
};
template<int BM, int BN, int WM, int WN, int SplitK>
struct opus_gemm_mxscale_bpreshuffle_narrow_fine_traits
: opus_gemm_small_lds_traits_gfx950<BM, BN, WM, WN, 4, 1, 2,
false, false, true, true, true, SplitK, 16, 128, 2, 0, true> {
static_assert((BM == 48 && BN == 64 && WM == 1 && WN == 4) ||
((BM == 64 || BM == 96) && BN == 128 && WM == 2 && WN == 2));
static_assert(SplitK == 1 || SplitK == 2);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=128;
static constexpr int B_N=128;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 128 && B_N == 128 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static_assert(E_M == 4 && E_N == 4 && E_K == 1);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static_assert(MAX_K_TILES == 128);
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 3;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static_assert(NUM_STAGES == 3 && MATRIX_LDS_BYTES == 101376);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;
static_assert(A_VMEM_INSTRUCTIONS == 4 && B_VMEM_INSTRUCTIONS == 4);
static_assert(VMEM_INSTRUCTIONS_PER_TILE == 8);
static constexpr int SCALE_PANEL = 32;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static_assert(SCALE_PANEL == 32 && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(B_M % VEC_SCALE_A == 0 && SFA_BYTES % VEC_SCALE_A == 0);
static_assert(SFA_PASSES == 1 && SFB_BYTES <= BLOCK_SIZE);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(B_SCALE_PACKS == 1);
static_assert(LDS_BYTES == 105504 && LDS_BYTES <= 160 * 1024);
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(C_LDS_ROW_STRIDE_ELEMS == 136);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0>
struct opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=512;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M=4;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert((B_M == 128 || B_M == 192) && (B_N == 128 || B_N == 256));
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr int SWIZZLE_GROUP_M = 4;
static constexpr int SWIZZLE_GROUP_N = 8;
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = 2048;
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = ScalePanel * GROUP_K;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int MATRIX_VMEM_INSTRUCTIONS =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static_assert(NUM_STAGES == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = SCALE_PANEL < 32 ? SCALE_PANEL : 32;
static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
using opus_gemm_mxscale_bpreshuffle_8wave_192x256_traits_gfx950 = opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<>;
struct opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=160;
static constexpr int B_N=128;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 160 && B_N == 128 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static_assert(E_M == 5 && E_N == 4 && E_K == 1);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static_assert(MAX_K_TILES == 128);
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 76032);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;
static_assert(A_VMEM_INSTRUCTIONS == 5 && B_VMEM_INSTRUCTIONS == 4);
static_assert(VMEM_INSTRUCTIONS_PER_TILE == 9);
static constexpr int SCALE_PANEL = 32;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static_assert(SCALE_PANEL == 32 && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(B_M % VEC_SCALE_A == 0 && SFA_BYTES % VEC_SCALE_A == 0);
static_assert(SFA_PASSES == 2 && SFB_BYTES <= BLOCK_SIZE);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(B_SCALE_PACKS == 1);
static_assert(LDS_BYTES == 81184 && LDS_BYTES <= 80 * 1024);
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(C_LDS_ROW_STRIDE_ELEMS == 136);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
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
template<int BM, int BN, int WaveK>
struct opus_gemm_mxscale_bpreshuffle_register_runtime_traits
: opus_gemm_small_register_traits_gfx950<BM, BN, 1, 1, 3, WaveK, 3, 3> {
static constexpr int REDUCE_VEC = 16, REDUCE_BLOCK = 128;
static_assert(WaveK == 1 || WaveK == 2);
static_assert((BM == 16 && (BN == 16 || BN == 32)) ||
(BM == 32 && (BN == 32 || BN == 64)));
};
template<int TileM>
struct opus_gemm_mxscale_bpreshuffle_2wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=128;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 2;
static constexpr int B_M = TileM;
static constexpr int B_N=64;
static constexpr int B_K=128;
static constexpr int T_M=1;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = 32;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static constexpr int B_SCALE_PACKS = (B_N + GROUP_N - 1) / GROUP_N;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(B_M == 16 || B_M == 32);
static_assert(NUM_WAVES == T_M * T_N * T_K && NUM_WAVES == 2);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M % (BLOCK_SIZE / (B_K / VEC_A)) == 0);
static_assert(B_N % (NUM_WAVES * W_N) == 0);
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(E_M <= 2 && E_N == 2 && E_K == 1);
static_assert(NUM_STAGES == 2 && B_K == GROUP_K && B_N <= GROUP_N);
static_assert(SFA_PASSES == 1 && B_SCALE_PACKS == 1);
static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(LDS_BYTES == (B_M == 16 ? 21664 : 26400));
static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * MIN_WGS_PER_CU <= 160 * 1024);
};
using opus_gemm_mxscale_bpreshuffle_2wave_16x64_traits_gfx950 =
opus_gemm_mxscale_bpreshuffle_2wave_traits_gfx950<16>;
struct opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950 {
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192;
static constexpr int B_N=256;
static constexpr int B_K=128;
static constexpr int T_M=4;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 192 && B_N == 256 && B_K == 128);
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr int SWIZZLE_GROUP_M = 4;
static constexpr int SWIZZLE_MAX_N = 2048;
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = 128;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SCALE_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(SFA_BYTES + BLOCK_SIZE * VEC_SCALE_A - 1) / (BLOCK_SIZE * VEC_SCALE_A);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M % VEC_SCALE_A == 0);
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 8 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SCALE_PANEL == 128 && SCALE_PANEL * B_K == MAX_K);
static_assert(SFA_BYTES == 24576 && SFA_PASSES == 3 && SFB_BYTES == 512);
static_assert(B_SCALE_PACKS == 1);
static_assert(LDS_BYTES == 143360 && LDS_BYTES <= 160 * 1024);
};
template<int FixedK>
struct opus_gemm_mxscale_bpreshuffle_shortk_traits
: opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
static constexpr int FIXED_K = FixedK;
static constexpr int SCALE_PANEL = 8;
static constexpr int MAX_K = FIXED_K;
static constexpr int MAX_K_TILES = FIXED_K / B_K;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(FIXED_K == 384 || FIXED_K == 768);
static_assert(MAX_K_TILES == 3 || MAX_K_TILES == 6);
static_assert(MAX_K_TILES <= SCALE_PANEL && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(SFA_BYTES == 1280 && SFB_BYTES == 8 && SFA_PASSES == 1);
static_assert(SFA_VECTORS_PER_GROUP == 10 && B_M % VEC_SCALE_A == 0);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 76032 && LDS_BYTES == 77320);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr bool PAD_M = true;
};
struct opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950 {
static constexpr int LOOP_UNROLL = 4;
};
template<int Waves, int ScalePanel = 64>
struct opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950 {
static_assert(Waves == 4 || Waves == 8);
static_assert(ScalePanel == 64 || ScalePanel == 128);
static constexpr int BLOCK_SIZE = Waves * 64;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = Waves;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M = Waves / 2, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % Waves == 0 && smem_n_rep % Waves == 0);
static_assert(E_N == 8 && C_REGS <= 256);
static_assert(LDS_BYTES <= 160 * 1024);
};
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
template<int ScalePanel = 32, int FixedK = 0, int GroupM = 0>
struct opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int FIXED_K = FixedK;
static constexpr int BLOCK_GROUP_M = GroupM;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=64;
static constexpr int B_N=64;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 64 && B_N == 64 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_K_VECTORS = B_K / VEC_A;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / A_K_VECTORS;
static constexpr int A_ROWS_PER_BLOCK = BLOCK_SIZE / A_K_VECTORS;
static constexpr int A_LOAD_PASSES = B_M / A_ROWS_PER_BLOCK;
static constexpr int A_LANES_K = WARP_SIZE / W_M;
static constexpr int A_CHUNK_STRIDE = A_LANES_K * VEC_A;
static_assert(E_M == 2 && E_K == 1);
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(A_K_VECTORS == 8 && A_LANES_K == 4 && A_ROWS_PER_WAVE == W_M / T_M);
static_assert(B_M % A_ROWS_PER_BLOCK == 0);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static_assert(B_N <= GROUP_N && B_K == GROUP_K);
static constexpr int SWIZZLE_PAIR_B_N = 64;
static constexpr unsigned SWIZZLE_MAX_GRID_N = 16;
static constexpr unsigned SWIZZLE_PAIR_MIN_GRID_M = 16;
static constexpr unsigned SWIZZLE_PAIR_MAX_GRID_M = 32;
static constexpr unsigned SWIZZLE_N_PARTITIONS = 4;
static constexpr unsigned SWIZZLE_M_PAIR = 2;
static constexpr unsigned SWIZZLE_PAIR_PARTITIONS = SWIZZLE_N_PARTITIONS * SWIZZLE_M_PAIR;
static constexpr int SWIZZLE_PAIR_SHIFT = 3;
static constexpr unsigned SWIZZLE_MIN_TILES = 256;
static constexpr unsigned SWIZZLE_SMALL_GRID_N = 8;
static constexpr unsigned SWIZZLE_SMALL_TILES = 768;
static constexpr unsigned SWIZZLE_SHORT_GRID_M = 24;
static constexpr unsigned SWIZZLE_LARGE_PARTITIONS = 16;
static_assert(SWIZZLE_PAIR_PARTITIONS == (1u << SWIZZLE_PAIR_SHIFT));
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 4;
static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
static_assert(NUM_STAGES == 4 && VEC_A == VEC_B);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_PRODUCERS_PER_GROUP;
static constexpr int SFA_WORDS = VEC_SCALE_A / sizeof(opus::u32_t);
static constexpr int VEC_SCALE_PACK_A = VEC_SCALE_A / sizeof(opus::u16_t);
static constexpr int SFA_PACK_CHUNKS = VEC_SCALE_A / VEC_SCALE_PACK_A;
static constexpr int SFA_WORDS_PER_CHUNK = SFA_WORDS / SFA_PACK_CHUNKS;
static constexpr unsigned SFA_PACK_PERM_LO = 0x05010400u;
static constexpr unsigned SFA_PACK_PERM_HI = 0x07030602u;
static constexpr int B_SCALE_PACKS = (B_N + GROUP_N - 1) / GROUP_N;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= SCALE_PANEL * B_K));
static_assert(SFA_PRODUCERS_PER_GROUP == 2 && SFA_PACK_CHUNKS == 2 && SFA_WORDS_PER_CHUNK == 2);
static_assert(WARP_SIZE % SFA_PRODUCERS_PER_GROUP == 0);
static_assert(SFA_ROWS_PER_REPEAT * E_M == B_M && E_M == sizeof(opus::u16_t));
static_assert(B_SCALE_PACKS == 1);
static_assert(SCALE_PANEL <= NUM_WAVES * SFA_K_COLUMNS_PER_WAVE && SCALE_PANEL <= BLOCK_SIZE);
static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * 2 <= 160 * 1024);
};
using opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_gfx950 = opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<>;
template<int BM,int Stages,int Panel,int FixedK=0>
struct opus_gemm_mxscale_bpreshuffle_geometry_traits : opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
static constexpr int B_M=BM, HALF_B_M=BM, FIXED_K=FixedK;
static constexpr int E_M=B_M/(T_M*W_M);
static constexpr int smem_m_rep=B_M*B_K/smem_linear_wave;
static constexpr int A_STAGE=smem_m_rep*(smem_linear_wave+smem_padding);
static constexpr int NUM_STAGES=Stages;
static constexpr int MATRIX_LDS_BYTES=NUM_STAGES*(A_STAGE+B_STAGE);
static constexpr int A_VMEM_INSTRUCTIONS=B_M*B_K/(BLOCK_SIZE*VEC_A);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE=A_VMEM_INSTRUCTIONS+B_VMEM_INSTRUCTIONS;
static constexpr int SCALE_PANEL=Panel;
static constexpr int SFA_BYTES=B_M*SCALE_PANEL;
static constexpr int SFB_BYTES=SCALE_PANEL;
static constexpr int SFA_VECTORS_PER_GROUP=B_M/VEC_SCALE_A;
static constexpr int SFA_PASSES=(SFA_BYTES+SFA_BYTES_PER_PASS-1)/SFA_BYTES_PER_PASS;
static constexpr int A_SCALE_PACKS=(E_M+3)/4;
static constexpr int LDS_BYTES=MATRIX_LDS_BYTES+SFA_BYTES+SFB_BYTES;
static constexpr int OUTPUT_PASSES=B_M*B_N/(BLOCK_SIZE*VEC_OUTPUT);
static_assert(BM==64||BM==96||BM==128||BM==160);
static_assert(Stages==2||Stages==3);
static_assert((Panel&(Panel-1))==0 && Panel>=8 && Panel<=32);
static_assert(FixedK==0 || (FixedK%128==0 && FixedK<=16384));
static_assert(B_M%(T_M*W_M)==0 && smem_m_rep%NUM_WAVES==0);
static_assert(B_M*B_N%(BLOCK_SIZE*VEC_OUTPUT)==0);
static_assert(B_M*C_LDS_ROW_STRIDE_ELEMS*2<=LDS_BYTES && LDS_BYTES<=160*1024);
};
template<int K,int Panel,bool Reset=false>
struct opus_gemm_mxscale_bpreshuffle_pin_fixed_traits : opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr int FIXED_K=K;
static constexpr bool RESET_SFA_BEFORE_LOAD=Reset;
static constexpr int SCALE_PANEL_K_CAPACITY=Panel;
static constexpr int SFA_PASSES_PER_CACHE_PANEL=Panel/SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_BYTES=B_M*Panel;
static constexpr int SFB_PANEL_BYTES=SCALE_N_HALVES*Panel*VEC_SF;
static constexpr int LDS_BYTES=SMEM_A_ELEMS+SMEM_B_ELEMS+SFA_PANEL_BYTES+SFB_PANEL_BYTES;
static_assert(K%128==0 && K<=16384 && Panel>=16 && Panel<=64);
static_assert((Panel&(Panel-1))==0 && SFA_PASSES_PER_CACHE_PANEL>0);
};
template<int K> struct opus_gemm_mxscale_bpreshuffle_pad_pin_fixed_traits : opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950 {
static constexpr int FIXED_K=K;
};
template<int NumStages = 3, int ScalePanel = 32, int FixedK = 0>
struct opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=64;
static constexpr int B_N=128;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 64 && B_N == 128 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_K_VECTORS = B_K / VEC_A;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / A_K_VECTORS;
static constexpr int A_ROWS_PER_BLOCK = BLOCK_SIZE / A_K_VECTORS;
static constexpr int A_LOAD_PASSES = B_M / A_ROWS_PER_BLOCK;
static constexpr int A_LANES_K = WARP_SIZE / W_M;
static constexpr int A_CHUNK_STRIDE = A_LANES_K * VEC_A;
static_assert(E_M == 2 && E_K == 1);
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(A_K_VECTORS == 8 && A_LANES_K == 4 && A_ROWS_PER_WAVE == W_M / T_M);
static_assert(B_M % A_ROWS_PER_BLOCK == 0);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static_assert(B_N <= GROUP_N && B_K == GROUP_K);
static constexpr int SWIZZLE_PAIR_B_N = 64;
static constexpr unsigned SWIZZLE_MAX_GRID_N = 16;
static constexpr unsigned SWIZZLE_PAIR_MIN_GRID_M = 16;
static constexpr unsigned SWIZZLE_PAIR_MAX_GRID_M = 32;
static constexpr unsigned SWIZZLE_N_PARTITIONS = 4;
static constexpr unsigned SWIZZLE_M_PAIR = 2;
static constexpr unsigned SWIZZLE_PAIR_PARTITIONS = SWIZZLE_N_PARTITIONS * SWIZZLE_M_PAIR;
static constexpr int SWIZZLE_PAIR_SHIFT = 3;
static constexpr unsigned SWIZZLE_MIN_TILES = 256;
static constexpr unsigned SWIZZLE_SMALL_GRID_N = 8;
static constexpr unsigned SWIZZLE_SMALL_TILES = 768;
static constexpr unsigned SWIZZLE_SHORT_GRID_M = 24;
static constexpr unsigned SWIZZLE_LARGE_PARTITIONS = 16;
static_assert(SWIZZLE_PAIR_PARTITIONS == (1u << SWIZZLE_PAIR_SHIFT));
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = NumStages;
static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
static_assert(NUM_STAGES >= 2 && NUM_STAGES <= 4 && VEC_A == VEC_B);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_PRODUCERS_PER_GROUP;
static constexpr int SFA_WORDS = VEC_SCALE_A / sizeof(opus::u32_t);
static constexpr int VEC_SCALE_PACK_A = VEC_SCALE_A / sizeof(opus::u16_t);
static constexpr int SFA_PACK_CHUNKS = VEC_SCALE_A / VEC_SCALE_PACK_A;
static constexpr int SFA_WORDS_PER_CHUNK = SFA_WORDS / SFA_PACK_CHUNKS;
static constexpr unsigned SFA_PACK_PERM_LO = 0x05010400u;
static constexpr unsigned SFA_PACK_PERM_HI = 0x07030602u;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= SCALE_PANEL * B_K));
static_assert(SFA_PRODUCERS_PER_GROUP == 2 && SFA_PACK_CHUNKS == 2 && SFA_WORDS_PER_CHUNK == 2);
static_assert(WARP_SIZE % SFA_PRODUCERS_PER_GROUP == 0);
static_assert(SFA_ROWS_PER_REPEAT * E_M == B_M && E_M == sizeof(opus::u16_t));
static_assert(B_SCALE_PACKS == 1);
static_assert(SCALE_PANEL <= NUM_WAVES * SFA_K_COLUMNS_PER_WAVE && SCALE_PANEL <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};
using opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_gfx950 = opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950<>;
using opus_gemm_mxscale_bpreshuffle_2wave_32x64_traits_gfx950 =
opus_gemm_mxscale_bpreshuffle_2wave_traits_gfx950<32>;
template<int BlockM, int WaveM, int WaveN, int Stages, int Cluster, int SplitK,
int ReduceVec=4, int ReduceBlock=128, int StoreCache=0, int FixedK=0>
struct opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950
: opus_gemm_small_lds_traits_gfx950<
BlockM, 128, WaveM, WaveN, Stages, Cluster, 2,
false, false, true, true, true,
SplitK, ReduceVec, ReduceBlock, StoreCache, FixedK, true> {};
namespace opus_bpreshuffle_policy {
template<class Base> constexpr int fixed_k() {
if constexpr (requires { Base::FIXED_K; }) return Base::FIXED_K;
else return 0;
}
template<class Base> constexpr int block_group_m() {
if constexpr (requires { Base::BLOCK_GROUP_M; }) return Base::BLOCK_GROUP_M;
else return 0;
}
}
template<class Base, int Schedule>
struct opus_gemm_mxscale_bpreshuffle_pipeline_traits : Base {
static constexpr int SCHEDULE = Schedule;
static constexpr int FIXED_K = opus_bpreshuffle_policy::fixed_k<Base>();
static constexpr int BLOCK_GROUP_M = opus_bpreshuffle_policy::block_group_m<Base>();
static constexpr int MIN_WGS_PER_CU = 1;
};

template<int Panel, int FixedK, bool Reset, int Unroll>
struct opus_gemm_mxscale_bpreshuffle_pin_config_traits
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr int FIXED_K = FixedK, LOOP_UNROLL = Unroll;
static constexpr bool RESET_SFA_BEFORE_LOAD = Reset;
static constexpr int SCALE_PANEL_K_CAPACITY = Panel;
static constexpr int SFA_PASSES_PER_CACHE_PANEL = Panel / SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_BYTES = B_M * Panel;
static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * Panel * VEC_SF;
static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
static_assert(Panel == 16 || Panel == 32 || Panel == 64);
static_assert(Unroll == 2 || Unroll == 4);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK % 128 == 0 && FixedK <= 16384));
static_assert(LDS_BYTES <= 160 * 1024);
};
template<int BM, int Stages, int Panel, int FixedK, int Schedule>
struct opus_gemm_mxscale_bpreshuffle_small_config_traits
: std::conditional_t<Schedule == 1,
opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950,
opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950> {
using Base = std::conditional_t<Schedule == 1,
opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950,
opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950>;
static constexpr int B_M = BM, HALF_B_M = BM, FIXED_K = FixedK;
static constexpr int E_M = BM / (Base::T_M * Base::W_M);
static constexpr int smem_m_rep = BM * Base::B_K / Base::smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (Base::smem_linear_wave + Base::smem_padding);
static constexpr int NUM_STAGES = Stages;
static constexpr int MATRIX_LDS_BYTES = Stages * (A_STAGE + Base::B_STAGE);
static constexpr int A_VMEM_INSTRUCTIONS = BM * Base::B_K / (Base::BLOCK_SIZE * Base::VEC_A);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + Base::B_VMEM_INSTRUCTIONS;
static constexpr int SCALE_PANEL = Panel, SFA_BYTES = BM * Panel, SFB_BYTES = Panel;
static constexpr int SFA_VECTORS_PER_GROUP = BM / Base::VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + Base::SFA_BYTES_PER_PASS - 1) / Base::SFA_BYTES_PER_PASS;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int OUTPUT_PASSES = BM * Base::B_N / (Base::BLOCK_SIZE * Base::VEC_OUTPUT);
static_assert(BM == 64 || BM == 96 || BM == 128 || BM == 160);
static_assert(Schedule == 1 || Schedule == 2);
static_assert(Schedule != 1 || BM == 128);
static_assert(Stages == 2 || Stages == 3);
static_assert(Panel >= 8 && Panel <= 32 && (Panel & (Panel - 1)) == 0);
static_assert(BM * Base::C_LDS_ROW_STRIDE_ELEMS * 2 <= LDS_BYTES && LDS_BYTES <= 160 * 1024);
};
template<class Base, int Vec, int Block>
struct opus_gemm_mxscale_bpreshuffle_runtime_config_traits : Base {
static constexpr int REDUCE_VEC = Vec, REDUCE_BLOCK = Block;
static constexpr int lds_bytes(int k, int split_k) {
const int total = k / Base::B_K;
const int loops = total / split_k + (total % split_k != 0);
const int stages = loops < Base::NUM_STAGES ? loops : Base::NUM_STAGES;
return stages * (Base::A_STAGE + Base::B_STAGE) +
(Base::REGISTER_SCALES ? 0 : (Base::B_M + Base::B_GROUPS) * loops);
}
};
template<int Panel, int FixedK, bool DirectB, int ChunkRows>
struct opus_gemm_mxscale_bpreshuffle_large_config_traits
: opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950 {
static constexpr int FIXED_K = FixedK, MAX_K = Panel * 128;
static constexpr int SCALE_PANEL = Panel, SFA_BYTES = B_M * Panel;
static constexpr int SFB_BYTES = Panel * sizeof(unsigned);
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / Panel;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES = (B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static constexpr int NUM_STAGES = DirectB ? 3 : 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + (DirectB ? 0 : B_STAGE));
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int B_DIRECT_SETS = 2, C_CHUNK_ROWS = DirectB ? ChunkRows : B_M;
static constexpr int C_CHUNKS = B_M / C_CHUNK_ROWS;
static constexpr int C_CHUNK_BYTES = C_CHUNK_ROWS * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t);
static constexpr int CHUNK_OUTPUT_PASSES = C_CHUNK_ROWS * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static_assert(Panel == 16 || Panel == 32 || Panel == 64 || Panel == 128);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK % 128 == 0 && FixedK <= MAX_K));
static_assert(B_M % C_CHUNK_ROWS == 0 && C_CHUNK_ROWS % 16 == 0);
static_assert(C_CHUNK_ROWS * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(C_CHUNK_BYTES <= LDS_BYTES && LDS_BYTES <= 160 * 1024);
};
}

namespace opus_bpreshuffle_gfx950::tiled::schedule_2 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_0;
using namespace opus_bpreshuffle_gfx950::tiled::schedule_1;
// SPDX-License-Identifier: Apache-2.0
}

namespace opus_bpreshuffle_gfx950::tiled::schedule_3 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_0;
using namespace opus_bpreshuffle_gfx950::tiled::schedule_1;
// SPDX-License-Identifier: Apache-2.0
}

namespace opus_bpreshuffle_gfx950::tiled::schedule_4 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_0;
using namespace opus_bpreshuffle_gfx950::tiled::schedule_1;
// SPDX-License-Identifier: Apache-2.0
}

namespace opus_bpreshuffle_gfx950::tiled::schedule_5 {
// SPDX-License-Identifier: Apache-2.0

struct opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int VEC_SF = 4;
static constexpr int VEC_SF_PAIR = 2 * VEC_SF;
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 : opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int LOOP_UNROLL = 2;
static constexpr bool RESET_SFA_BEFORE_LOAD = false;
static constexpr int B_M=256;
static constexpr int B_N=256;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M / 2;
static constexpr int HALF_B_N = B_N / 2;
static_assert(NUM_WAVES == 4);
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(HALF_B_M % (W_M * T_M) == 0);
static_assert(HALF_B_N % (W_N * T_N) == 0);
static_assert(B_K % (W_K * T_K) == 0);
static constexpr int E_M = HALF_B_M / (W_M * T_M);
static constexpr int E_N = HALF_B_N / (W_N * T_N);
static constexpr int E_K = B_K / (W_K * T_K);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int OUTPUT_TILES_PER_WG = 1;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_SCALE_SF = 4;
static constexpr int VEC_SCALE_SF_PAIR = 8;
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MFMA_SCALE_GROUP_K = 32;
static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_sub = smem_linear_wave / B_K;
static constexpr int smem_m_rep = HALF_B_M / smem_sub;
static constexpr int smem_n_rep = HALF_B_N / smem_sub;
static constexpr int smem_padding = 32;
static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
static_assert(NUM_KGROUPS == 4);
static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int a_ds_read_insts = E_M * W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SCALE_PANEL_K_CAPACITY = 64;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M;
static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_PITCH = B_M;
static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;
static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
static_assert(E_M == 4 && E_N == 4 && E_K == 1);
static_assert(VEC_SCALE_SF == E_M);
static_assert(VEC_SCALE_SF_PAIR == 2 * VEC_SCALE_SF);
static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS == SCALE_PANEL_K_CAPACITY);
static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_scale_reset_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr bool RESET_SFA_BEFORE_LOAD = true;
};

__host__ __device__ inline int opus_mxscale_bpreshuffle_ceil_div(int a, int b) {
return (a + b - 1) / b;
}
}

#endif

#if defined(OPUS_BP_INCLUDE_TILED_69) && !defined(OPUS_BP_INCLUDE_TILED_69_DONE)
#define OPUS_BP_INCLUDE_TILED_69_DONE
namespace opus_bpreshuffle_gfx950::tiled::schedule_69 {
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

struct opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int VEC_SF = 4;
static constexpr int VEC_SF_PAIR = 2 * VEC_SF;
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 : opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int LOOP_UNROLL = 2;
static constexpr bool RESET_SFA_BEFORE_LOAD = false;
static constexpr int B_M=256;
static constexpr int B_N=256;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M / 2;
static constexpr int HALF_B_N = B_N / 2;
static_assert(NUM_WAVES == 4);
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(HALF_B_M % (W_M * T_M) == 0);
static_assert(HALF_B_N % (W_N * T_N) == 0);
static_assert(B_K % (W_K * T_K) == 0);
static constexpr int E_M = HALF_B_M / (W_M * T_M);
static constexpr int E_N = HALF_B_N / (W_N * T_N);
static constexpr int E_K = B_K / (W_K * T_K);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int OUTPUT_TILES_PER_WG = 1;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_SCALE_SF = 4;
static constexpr int VEC_SCALE_SF_PAIR = 8;
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MFMA_SCALE_GROUP_K = 32;
static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_sub = smem_linear_wave / B_K;
static constexpr int smem_m_rep = HALF_B_M / smem_sub;
static constexpr int smem_n_rep = HALF_B_N / smem_sub;
static constexpr int smem_padding = 32;
static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
static_assert(NUM_KGROUPS == 4);
static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int a_ds_read_insts = E_M * W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SCALE_PANEL_K_CAPACITY = 64;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M;
static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_PITCH = B_M;
static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;
static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
static_assert(E_M == 4 && E_N == 4 && E_K == 1);
static_assert(VEC_SCALE_SF == E_M);
static_assert(VEC_SCALE_SF_PAIR == 2 * VEC_SCALE_SF);
static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS == SCALE_PANEL_K_CAPACITY);
static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_scale_reset_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr bool RESET_SFA_BEFORE_LOAD = true;
};

__host__ __device__ inline int opus_mxscale_bpreshuffle_ceil_div(int a, int b) {
return (a + b - 1) / b;
}
template<int BlockM, int BlockN, int WaveM, int WaveN, int Prefetch,
int WaveK = 1, int Output = 0, int BCache = 0,
int FixedK = 0, bool PadN = false, bool ReuseBScale = false>
struct opus_gemm_small_register_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K = WaveK;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * WaveK * 64;
static constexpr int E_M = BlockM / (WaveM * W_M);
static constexpr int E_N = BlockN / (WaveN * W_N);
static constexpr int PREFETCH = Prefetch, WAVE_K = WaveK;
static constexpr int B_CACHE = BCache, OUTPUT = Output;
static constexpr int FIXED_K = FixedK;
static constexpr bool N_TAIL = PadN, REUSE_B_SCALE = ReuseBScale;
static_assert(BlockM % (WaveM * W_M) == 0 && BlockN % (WaveN * W_N) == 0);
static_assert(E_M > 0 && E_M <= 4 && E_N > 0 && Prefetch > 0);
static_assert(BLOCK_SIZE <= 512 && WaveK > 0);
static_assert(Output == 0 || Output == 3 || Output == 4);
static_assert(BCache >= 0 && BCache <= 3);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % 128 == 0));
static_assert(!PadN || Output == 4);
static_assert(!ReuseBScale || 128 % BlockN == 0);
static_assert(Output != 4 || WaveK > 1);
static_assert((WaveK - (Output == 4 ? 0 : 1)) * BlockM * BlockN * sizeof(float) <= 160 * 1024);
};
template<int BlockM, int BlockN, int WaveM, int WaveN,
int Stages, int Cluster = 1, int Output = 2,
bool RegisterScales = false, bool XorLds = false, bool EarlyScaleLoads = false,
bool PrefetchBeforeRead = false, bool ReadOnlyDrain = false,
int SplitK = 1, int ReduceVec = 4, int ReduceBlock = 128,
int StoreCache = 0, int FixedK = 0, bool FineMLoads = false>
struct opus_gemm_small_lds_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * 64;
static constexpr int WARP_SIZE = 64, NUM_WAVES = WaveM * WaveN;
static constexpr int E_M = BlockM / (WaveM * 16), E_N = BlockN / (WaveN * 16), E_K = 1;
static constexpr int HALF_B_M = BlockM, HALF_B_N = BlockN;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_SCALE_A = 16;
static constexpr int smem_linear_wave = 1024, smem_padding = XorLds ? 0 : 32;
static constexpr int smem_m_rep = BlockM / 8, smem_n_rep = BlockN / 8;
static constexpr int A_STAGE = BlockM / 8 * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = BlockN / 8 * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = Stages, CLUSTER = Cluster;
static constexpr int OUTPUT = Output;
static constexpr bool REGISTER_SCALES = RegisterScales, XOR_LDS = XorLds;
static constexpr bool EARLY_SCALE_LOADS = EarlyScaleLoads;
static constexpr bool PREFETCH_BEFORE_READ = PrefetchBeforeRead;
static constexpr bool READ_ONLY_DRAIN = ReadOnlyDrain;
static constexpr bool FINE_M_LOADS = FineMLoads;
static constexpr int SPLIT_K = SplitK, REDUCE_VEC = ReduceVec, REDUCE_BLOCK = ReduceBlock;
static constexpr int STORE_CACHE = StoreCache, FIXED_K = FixedK;
static constexpr int MAX_K = 16384, MAX_LOOPS = (MAX_K / B_K + SplitK - 1) / SplitK;
static constexpr int B_GROUPS = (BlockN + 127) / 128;
static constexpr int MAX_SFA_BYTES = BlockM * MAX_LOOPS;
static constexpr int MAX_LDS_BYTES = Stages * (A_STAGE + B_STAGE) +
(RegisterScales ? 0 : (BlockM + B_GROUPS) * MAX_LOOPS);
static constexpr int VMEM_TILE = (BlockM + BlockN) * 128 / (BLOCK_SIZE * 16) +
(RegisterScales ? E_M + B_GROUPS : 0);
static_assert(BlockM % (T_M * 16) == 0 && BlockN % (NUM_WAVES * 16) == 0);
static_assert(FineMLoads || BlockM % (BLOCK_SIZE / 8) == 0);
static_assert(E_M > 0 && E_M <= (FineMLoads ? 8 : 4) && E_N > 0 && MAX_LOOPS >= Stages);
static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4 || SplitK == 8);
static_assert(ReduceVec >= 4 && ReduceVec % 4 == 0 && ReduceVec <= 16);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= MAX_K && FixedK % 128 == 0));
static_assert(Cluster > 0 && Stages % Cluster == 0 && Stages >= 2 * Cluster);
static_assert(MAX_LDS_BYTES <= 160 * 1024 && B_GROUPS * MAX_LOOPS <= BLOCK_SIZE);
static_assert(Output == 1 || Output == 2);
static_assert(!XorLds || (NUM_WAVES == 4 && A_STAGE % 4096 == 0 && B_STAGE % 4096 == 0));
static_assert(Output != 1 || (BlockM * (BlockN + 8) * 2 <= A_STAGE + B_STAGE &&
BlockM * BlockN % (BLOCK_SIZE * 8) == 0));
static constexpr int lds_bytes(int k) {
const int loops = (k / B_K + SplitK - 1) / SplitK;
const int stages = loops < NUM_STAGES ? loops : NUM_STAGES;
return stages * (A_STAGE + B_STAGE) + (RegisterScales ? 0 : (BlockM + B_GROUPS) * loops);
}
};
using opus_gemm_mxscale_bpreshuffle_small_register_16x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 6, 1, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_16x16_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 16, 1, 1, 2, 8>;
using opus_gemm_mxscale_bpreshuffle_small_register_32x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<32, 32, 1, 1, 3, 4, 3, 0, 0, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 1, 4, 8, 2, 2, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_64x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 4, 1, 2, false, false, true, true, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_96x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<96, 64, 2, 2, 4, 1, 1, false, false, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_64x128_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<64, 128, 4, 2, 6, 2>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 2, 2, 4, 1, 2, true>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 2, 2, 4, 1, 2, true, true>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x128_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 128, 2, 2, 4, 2, 2, true, true, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x16_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 16, 1, 1, 3, 8, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 3, 4, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_16x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 2, 8, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<32, 32, 1, 1, 2, 8, 3, 3, 0, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x64_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<32, 64, 1, 1, 2, 4, 3, 3, 0, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_deep_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 1, 4, 12, 4, 2, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_deep_64x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 8, 2, 2, false, false, true>;
constexpr int opus_gemm_mxscale_bpreshuffle_small_gcd(int a, int b) { while (b) { int r=a%b; a=b; b=r; } return a; }
template<class Baseline, int Ahead>
struct opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits : Baseline {
static constexpr int B_AHEAD = Ahead, B_SLOTS = Ahead + 1;
static constexpr int RING_PERIOD = Baseline::NUM_STAGES * B_SLOTS / opus_gemm_mxscale_bpreshuffle_small_gcd(Baseline::NUM_STAGES, B_SLOTS);
static constexpr int B_STAGE = 0;
static constexpr int C_BYTES = Baseline::OUTPUT == 1 ? Baseline::B_M * (Baseline::B_N + 8) * 2 : 0;
static constexpr int MAX_LDS_BYTES = Baseline::NUM_STAGES * Baseline::A_STAGE +
(Baseline::REGISTER_SCALES ? 0 : (Baseline::B_M + Baseline::B_GROUPS) * Baseline::MAX_LOOPS);
static_assert(Ahead >= 1 && Ahead <= 3 && Baseline::T_K == 1);
static_assert(MAX_LDS_BYTES <= 160 * 1024 && C_BYTES <= 160 * 1024);
static constexpr int lds_bytes(int k) {
const int loops = (k / Baseline::B_K + Baseline::SPLIT_K - 1) / Baseline::SPLIT_K;
const int stages = loops < Baseline::NUM_STAGES ? loops : Baseline::NUM_STAGES;
const int matrix_scale = stages * Baseline::A_STAGE +
(Baseline::REGISTER_SCALES ? 0 : (Baseline::B_M + Baseline::B_GROUPS) * loops);
return matrix_scale > C_BYTES ? matrix_scale : C_BYTES;
}
};
template<int Actual> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits;
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9043> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9044> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_64x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9045> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_96x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9046> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_64x128_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9047> { using type = opus_gemm_mxscale_bpreshuffle_small_regscale_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9049> { using type = opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x128_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9055> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_deep_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9056> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_deep_64x64_traits_gfx950; };
template<int Actual, int Ahead> using opus_gemm_mxscale_bpreshuffle_small_direct_b_traits = opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits<typename opus_gemm_mxscale_bpreshuffle_small_baseline_traits<Actual>::type, Ahead>;
template<int BM, int BN, int WaveK>
struct opus_gemm_mxscale_bpreshuffle_register_split_traits
: opus_gemm_small_register_traits_gfx950<BM, BN, 1, 1, 3, WaveK, 3, 3> {
static constexpr int GLOBAL_SPLIT_K = 4;
static_assert(WaveK == 1 || WaveK == 2);
static_assert((BM == 16 && (BN == 16 || BN == 32)) ||
(BM == 32 && (BN == 32 || BN == 64)));
};
template<int BM, int BN, int WM, int WN, int SplitK>
struct opus_gemm_mxscale_bpreshuffle_narrow_fine_traits
: opus_gemm_small_lds_traits_gfx950<BM, BN, WM, WN, 4, 1, 2,
false, false, true, true, true, SplitK, 16, 128, 2, 0, true> {
static_assert((BM == 48 && BN == 64 && WM == 1 && WN == 4) ||
((BM == 64 || BM == 96) && BN == 128 && WM == 2 && WN == 2));
static_assert(SplitK == 1 || SplitK == 2);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=128;
static constexpr int B_N=128;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 128 && B_N == 128 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static_assert(E_M == 4 && E_N == 4 && E_K == 1);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static_assert(MAX_K_TILES == 128);
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 3;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static_assert(NUM_STAGES == 3 && MATRIX_LDS_BYTES == 101376);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;
static_assert(A_VMEM_INSTRUCTIONS == 4 && B_VMEM_INSTRUCTIONS == 4);
static_assert(VMEM_INSTRUCTIONS_PER_TILE == 8);
static constexpr int SCALE_PANEL = 32;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static_assert(SCALE_PANEL == 32 && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(B_M % VEC_SCALE_A == 0 && SFA_BYTES % VEC_SCALE_A == 0);
static_assert(SFA_PASSES == 1 && SFB_BYTES <= BLOCK_SIZE);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(B_SCALE_PACKS == 1);
static_assert(LDS_BYTES == 105504 && LDS_BYTES <= 160 * 1024);
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(C_LDS_ROW_STRIDE_ELEMS == 136);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0>
struct opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=512;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M=4;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert((B_M == 128 || B_M == 192) && (B_N == 128 || B_N == 256));
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr int SWIZZLE_GROUP_M = 4;
static constexpr int SWIZZLE_GROUP_N = 8;
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = 2048;
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = ScalePanel * GROUP_K;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int MATRIX_VMEM_INSTRUCTIONS =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static_assert(NUM_STAGES == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = SCALE_PANEL < 32 ? SCALE_PANEL : 32;
static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
using opus_gemm_mxscale_bpreshuffle_8wave_192x256_traits_gfx950 = opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<>;
struct opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=160;
static constexpr int B_N=128;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 160 && B_N == 128 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static_assert(E_M == 5 && E_N == 4 && E_K == 1);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static_assert(MAX_K_TILES == 128);
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 76032);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;
static_assert(A_VMEM_INSTRUCTIONS == 5 && B_VMEM_INSTRUCTIONS == 4);
static_assert(VMEM_INSTRUCTIONS_PER_TILE == 9);
static constexpr int SCALE_PANEL = 32;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static_assert(SCALE_PANEL == 32 && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(B_M % VEC_SCALE_A == 0 && SFA_BYTES % VEC_SCALE_A == 0);
static_assert(SFA_PASSES == 2 && SFB_BYTES <= BLOCK_SIZE);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(B_SCALE_PACKS == 1);
static_assert(LDS_BYTES == 81184 && LDS_BYTES <= 80 * 1024);
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(C_LDS_ROW_STRIDE_ELEMS == 136);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
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
template<int BM, int BN, int WaveK>
struct opus_gemm_mxscale_bpreshuffle_register_runtime_traits
: opus_gemm_small_register_traits_gfx950<BM, BN, 1, 1, 3, WaveK, 3, 3> {
static constexpr int REDUCE_VEC = 16, REDUCE_BLOCK = 128;
static_assert(WaveK == 1 || WaveK == 2);
static_assert((BM == 16 && (BN == 16 || BN == 32)) ||
(BM == 32 && (BN == 32 || BN == 64)));
};
template<int TileM>
struct opus_gemm_mxscale_bpreshuffle_2wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=128;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 2;
static constexpr int B_M = TileM;
static constexpr int B_N=64;
static constexpr int B_K=128;
static constexpr int T_M=1;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = 32;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static constexpr int B_SCALE_PACKS = (B_N + GROUP_N - 1) / GROUP_N;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(B_M == 16 || B_M == 32);
static_assert(NUM_WAVES == T_M * T_N * T_K && NUM_WAVES == 2);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M % (BLOCK_SIZE / (B_K / VEC_A)) == 0);
static_assert(B_N % (NUM_WAVES * W_N) == 0);
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(E_M <= 2 && E_N == 2 && E_K == 1);
static_assert(NUM_STAGES == 2 && B_K == GROUP_K && B_N <= GROUP_N);
static_assert(SFA_PASSES == 1 && B_SCALE_PACKS == 1);
static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(LDS_BYTES == (B_M == 16 ? 21664 : 26400));
static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * MIN_WGS_PER_CU <= 160 * 1024);
};
using opus_gemm_mxscale_bpreshuffle_2wave_16x64_traits_gfx950 =
opus_gemm_mxscale_bpreshuffle_2wave_traits_gfx950<16>;
struct opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950 {
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192;
static constexpr int B_N=256;
static constexpr int B_K=128;
static constexpr int T_M=4;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 192 && B_N == 256 && B_K == 128);
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr int SWIZZLE_GROUP_M = 4;
static constexpr int SWIZZLE_MAX_N = 2048;
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = 128;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SCALE_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(SFA_BYTES + BLOCK_SIZE * VEC_SCALE_A - 1) / (BLOCK_SIZE * VEC_SCALE_A);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M % VEC_SCALE_A == 0);
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 8 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SCALE_PANEL == 128 && SCALE_PANEL * B_K == MAX_K);
static_assert(SFA_BYTES == 24576 && SFA_PASSES == 3 && SFB_BYTES == 512);
static_assert(B_SCALE_PACKS == 1);
static_assert(LDS_BYTES == 143360 && LDS_BYTES <= 160 * 1024);
};
template<int FixedK>
struct opus_gemm_mxscale_bpreshuffle_shortk_traits
: opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
static constexpr int FIXED_K = FixedK;
static constexpr int SCALE_PANEL = 8;
static constexpr int MAX_K = FIXED_K;
static constexpr int MAX_K_TILES = FIXED_K / B_K;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(FIXED_K == 384 || FIXED_K == 768);
static_assert(MAX_K_TILES == 3 || MAX_K_TILES == 6);
static_assert(MAX_K_TILES <= SCALE_PANEL && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(SFA_BYTES == 1280 && SFB_BYTES == 8 && SFA_PASSES == 1);
static_assert(SFA_VECTORS_PER_GROUP == 10 && B_M % VEC_SCALE_A == 0);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 76032 && LDS_BYTES == 77320);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr bool PAD_M = true;
};
struct opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950 {
static constexpr int LOOP_UNROLL = 4;
};
template<int Waves, int ScalePanel = 64>
struct opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950 {
static_assert(Waves == 4 || Waves == 8);
static_assert(ScalePanel == 64 || ScalePanel == 128);
static constexpr int BLOCK_SIZE = Waves * 64;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = Waves;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M = Waves / 2, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % Waves == 0 && smem_n_rep % Waves == 0);
static_assert(E_N == 8 && C_REGS <= 256);
static_assert(LDS_BYTES <= 160 * 1024);
};
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
template<int ScalePanel = 32, int FixedK = 0, int GroupM = 0>
struct opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int FIXED_K = FixedK;
static constexpr int BLOCK_GROUP_M = GroupM;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=64;
static constexpr int B_N=64;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 64 && B_N == 64 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_K_VECTORS = B_K / VEC_A;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / A_K_VECTORS;
static constexpr int A_ROWS_PER_BLOCK = BLOCK_SIZE / A_K_VECTORS;
static constexpr int A_LOAD_PASSES = B_M / A_ROWS_PER_BLOCK;
static constexpr int A_LANES_K = WARP_SIZE / W_M;
static constexpr int A_CHUNK_STRIDE = A_LANES_K * VEC_A;
static_assert(E_M == 2 && E_K == 1);
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(A_K_VECTORS == 8 && A_LANES_K == 4 && A_ROWS_PER_WAVE == W_M / T_M);
static_assert(B_M % A_ROWS_PER_BLOCK == 0);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static_assert(B_N <= GROUP_N && B_K == GROUP_K);
static constexpr int SWIZZLE_PAIR_B_N = 64;
static constexpr unsigned SWIZZLE_MAX_GRID_N = 16;
static constexpr unsigned SWIZZLE_PAIR_MIN_GRID_M = 16;
static constexpr unsigned SWIZZLE_PAIR_MAX_GRID_M = 32;
static constexpr unsigned SWIZZLE_N_PARTITIONS = 4;
static constexpr unsigned SWIZZLE_M_PAIR = 2;
static constexpr unsigned SWIZZLE_PAIR_PARTITIONS = SWIZZLE_N_PARTITIONS * SWIZZLE_M_PAIR;
static constexpr int SWIZZLE_PAIR_SHIFT = 3;
static constexpr unsigned SWIZZLE_MIN_TILES = 256;
static constexpr unsigned SWIZZLE_SMALL_GRID_N = 8;
static constexpr unsigned SWIZZLE_SMALL_TILES = 768;
static constexpr unsigned SWIZZLE_SHORT_GRID_M = 24;
static constexpr unsigned SWIZZLE_LARGE_PARTITIONS = 16;
static_assert(SWIZZLE_PAIR_PARTITIONS == (1u << SWIZZLE_PAIR_SHIFT));
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 4;
static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
static_assert(NUM_STAGES == 4 && VEC_A == VEC_B);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_PRODUCERS_PER_GROUP;
static constexpr int SFA_WORDS = VEC_SCALE_A / sizeof(opus::u32_t);
static constexpr int VEC_SCALE_PACK_A = VEC_SCALE_A / sizeof(opus::u16_t);
static constexpr int SFA_PACK_CHUNKS = VEC_SCALE_A / VEC_SCALE_PACK_A;
static constexpr int SFA_WORDS_PER_CHUNK = SFA_WORDS / SFA_PACK_CHUNKS;
static constexpr unsigned SFA_PACK_PERM_LO = 0x05010400u;
static constexpr unsigned SFA_PACK_PERM_HI = 0x07030602u;
static constexpr int B_SCALE_PACKS = (B_N + GROUP_N - 1) / GROUP_N;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= SCALE_PANEL * B_K));
static_assert(SFA_PRODUCERS_PER_GROUP == 2 && SFA_PACK_CHUNKS == 2 && SFA_WORDS_PER_CHUNK == 2);
static_assert(WARP_SIZE % SFA_PRODUCERS_PER_GROUP == 0);
static_assert(SFA_ROWS_PER_REPEAT * E_M == B_M && E_M == sizeof(opus::u16_t));
static_assert(B_SCALE_PACKS == 1);
static_assert(SCALE_PANEL <= NUM_WAVES * SFA_K_COLUMNS_PER_WAVE && SCALE_PANEL <= BLOCK_SIZE);
static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * 2 <= 160 * 1024);
};
using opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_gfx950 = opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<>;
template<int BM,int Stages,int Panel,int FixedK=0>
struct opus_gemm_mxscale_bpreshuffle_geometry_traits : opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
static constexpr int B_M=BM, HALF_B_M=BM, FIXED_K=FixedK;
static constexpr int E_M=B_M/(T_M*W_M);
static constexpr int smem_m_rep=B_M*B_K/smem_linear_wave;
static constexpr int A_STAGE=smem_m_rep*(smem_linear_wave+smem_padding);
static constexpr int NUM_STAGES=Stages;
static constexpr int MATRIX_LDS_BYTES=NUM_STAGES*(A_STAGE+B_STAGE);
static constexpr int A_VMEM_INSTRUCTIONS=B_M*B_K/(BLOCK_SIZE*VEC_A);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE=A_VMEM_INSTRUCTIONS+B_VMEM_INSTRUCTIONS;
static constexpr int SCALE_PANEL=Panel;
static constexpr int SFA_BYTES=B_M*SCALE_PANEL;
static constexpr int SFB_BYTES=SCALE_PANEL;
static constexpr int SFA_VECTORS_PER_GROUP=B_M/VEC_SCALE_A;
static constexpr int SFA_PASSES=(SFA_BYTES+SFA_BYTES_PER_PASS-1)/SFA_BYTES_PER_PASS;
static constexpr int A_SCALE_PACKS=(E_M+3)/4;
static constexpr int LDS_BYTES=MATRIX_LDS_BYTES+SFA_BYTES+SFB_BYTES;
static constexpr int OUTPUT_PASSES=B_M*B_N/(BLOCK_SIZE*VEC_OUTPUT);
static_assert(BM==64||BM==96||BM==128||BM==160);
static_assert(Stages==2||Stages==3);
static_assert((Panel&(Panel-1))==0 && Panel>=8 && Panel<=32);
static_assert(FixedK==0 || (FixedK%128==0 && FixedK<=16384));
static_assert(B_M%(T_M*W_M)==0 && smem_m_rep%NUM_WAVES==0);
static_assert(B_M*B_N%(BLOCK_SIZE*VEC_OUTPUT)==0);
static_assert(B_M*C_LDS_ROW_STRIDE_ELEMS*2<=LDS_BYTES && LDS_BYTES<=160*1024);
};
template<int K,int Panel,bool Reset=false>
struct opus_gemm_mxscale_bpreshuffle_pin_fixed_traits : opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr int FIXED_K=K;
static constexpr bool RESET_SFA_BEFORE_LOAD=Reset;
static constexpr int SCALE_PANEL_K_CAPACITY=Panel;
static constexpr int SFA_PASSES_PER_CACHE_PANEL=Panel/SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_BYTES=B_M*Panel;
static constexpr int SFB_PANEL_BYTES=SCALE_N_HALVES*Panel*VEC_SF;
static constexpr int LDS_BYTES=SMEM_A_ELEMS+SMEM_B_ELEMS+SFA_PANEL_BYTES+SFB_PANEL_BYTES;
static_assert(K%128==0 && K<=16384 && Panel>=16 && Panel<=64);
static_assert((Panel&(Panel-1))==0 && SFA_PASSES_PER_CACHE_PANEL>0);
};
template<int K> struct opus_gemm_mxscale_bpreshuffle_pad_pin_fixed_traits : opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950 {
static constexpr int FIXED_K=K;
};
template<int NumStages = 3, int ScalePanel = 32, int FixedK = 0>
struct opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=64;
static constexpr int B_N=128;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 64 && B_N == 128 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_K_VECTORS = B_K / VEC_A;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / A_K_VECTORS;
static constexpr int A_ROWS_PER_BLOCK = BLOCK_SIZE / A_K_VECTORS;
static constexpr int A_LOAD_PASSES = B_M / A_ROWS_PER_BLOCK;
static constexpr int A_LANES_K = WARP_SIZE / W_M;
static constexpr int A_CHUNK_STRIDE = A_LANES_K * VEC_A;
static_assert(E_M == 2 && E_K == 1);
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(A_K_VECTORS == 8 && A_LANES_K == 4 && A_ROWS_PER_WAVE == W_M / T_M);
static_assert(B_M % A_ROWS_PER_BLOCK == 0);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static_assert(B_N <= GROUP_N && B_K == GROUP_K);
static constexpr int SWIZZLE_PAIR_B_N = 64;
static constexpr unsigned SWIZZLE_MAX_GRID_N = 16;
static constexpr unsigned SWIZZLE_PAIR_MIN_GRID_M = 16;
static constexpr unsigned SWIZZLE_PAIR_MAX_GRID_M = 32;
static constexpr unsigned SWIZZLE_N_PARTITIONS = 4;
static constexpr unsigned SWIZZLE_M_PAIR = 2;
static constexpr unsigned SWIZZLE_PAIR_PARTITIONS = SWIZZLE_N_PARTITIONS * SWIZZLE_M_PAIR;
static constexpr int SWIZZLE_PAIR_SHIFT = 3;
static constexpr unsigned SWIZZLE_MIN_TILES = 256;
static constexpr unsigned SWIZZLE_SMALL_GRID_N = 8;
static constexpr unsigned SWIZZLE_SMALL_TILES = 768;
static constexpr unsigned SWIZZLE_SHORT_GRID_M = 24;
static constexpr unsigned SWIZZLE_LARGE_PARTITIONS = 16;
static_assert(SWIZZLE_PAIR_PARTITIONS == (1u << SWIZZLE_PAIR_SHIFT));
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = NumStages;
static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
static_assert(NUM_STAGES >= 2 && NUM_STAGES <= 4 && VEC_A == VEC_B);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_PRODUCERS_PER_GROUP;
static constexpr int SFA_WORDS = VEC_SCALE_A / sizeof(opus::u32_t);
static constexpr int VEC_SCALE_PACK_A = VEC_SCALE_A / sizeof(opus::u16_t);
static constexpr int SFA_PACK_CHUNKS = VEC_SCALE_A / VEC_SCALE_PACK_A;
static constexpr int SFA_WORDS_PER_CHUNK = SFA_WORDS / SFA_PACK_CHUNKS;
static constexpr unsigned SFA_PACK_PERM_LO = 0x05010400u;
static constexpr unsigned SFA_PACK_PERM_HI = 0x07030602u;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= SCALE_PANEL * B_K));
static_assert(SFA_PRODUCERS_PER_GROUP == 2 && SFA_PACK_CHUNKS == 2 && SFA_WORDS_PER_CHUNK == 2);
static_assert(WARP_SIZE % SFA_PRODUCERS_PER_GROUP == 0);
static_assert(SFA_ROWS_PER_REPEAT * E_M == B_M && E_M == sizeof(opus::u16_t));
static_assert(B_SCALE_PACKS == 1);
static_assert(SCALE_PANEL <= NUM_WAVES * SFA_K_COLUMNS_PER_WAVE && SCALE_PANEL <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};
using opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_gfx950 = opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950<>;
using opus_gemm_mxscale_bpreshuffle_2wave_32x64_traits_gfx950 =
opus_gemm_mxscale_bpreshuffle_2wave_traits_gfx950<32>;
template<int BlockM, int WaveM, int WaveN, int Stages, int Cluster, int SplitK,
int ReduceVec=4, int ReduceBlock=128, int StoreCache=0, int FixedK=0>
struct opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950
: opus_gemm_small_lds_traits_gfx950<
BlockM, 128, WaveM, WaveN, Stages, Cluster, 2,
false, false, true, true, true,
SplitK, ReduceVec, ReduceBlock, StoreCache, FixedK, true> {};
namespace opus_bpreshuffle_policy {
template<class Base> constexpr int fixed_k() {
if constexpr (requires { Base::FIXED_K; }) return Base::FIXED_K;
else return 0;
}
template<class Base> constexpr int block_group_m() {
if constexpr (requires { Base::BLOCK_GROUP_M; }) return Base::BLOCK_GROUP_M;
else return 0;
}
}
template<class Base, int Schedule>
struct opus_gemm_mxscale_bpreshuffle_pipeline_traits : Base {
static constexpr int SCHEDULE = Schedule;
static constexpr int FIXED_K = opus_bpreshuffle_policy::fixed_k<Base>();
static constexpr int BLOCK_GROUP_M = opus_bpreshuffle_policy::block_group_m<Base>();
static constexpr int MIN_WGS_PER_CU = 1;
};

template<int Panel, int FixedK, bool Reset, int Unroll>
struct opus_gemm_mxscale_bpreshuffle_pin_config_traits
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr int FIXED_K = FixedK, LOOP_UNROLL = Unroll;
static constexpr bool RESET_SFA_BEFORE_LOAD = Reset;
static constexpr int SCALE_PANEL_K_CAPACITY = Panel;
static constexpr int SFA_PASSES_PER_CACHE_PANEL = Panel / SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_BYTES = B_M * Panel;
static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * Panel * VEC_SF;
static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
static_assert(Panel == 16 || Panel == 32 || Panel == 64);
static_assert(Unroll == 2 || Unroll == 4);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK % 128 == 0 && FixedK <= 16384));
static_assert(LDS_BYTES <= 160 * 1024);
};
template<int BM, int Stages, int Panel, int FixedK, int Schedule>
struct opus_gemm_mxscale_bpreshuffle_small_config_traits
: std::conditional_t<Schedule == 1,
opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950,
opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950> {
using Base = std::conditional_t<Schedule == 1,
opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950,
opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950>;
static constexpr int B_M = BM, HALF_B_M = BM, FIXED_K = FixedK;
static constexpr int E_M = BM / (Base::T_M * Base::W_M);
static constexpr int smem_m_rep = BM * Base::B_K / Base::smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (Base::smem_linear_wave + Base::smem_padding);
static constexpr int NUM_STAGES = Stages;
static constexpr int MATRIX_LDS_BYTES = Stages * (A_STAGE + Base::B_STAGE);
static constexpr int A_VMEM_INSTRUCTIONS = BM * Base::B_K / (Base::BLOCK_SIZE * Base::VEC_A);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + Base::B_VMEM_INSTRUCTIONS;
static constexpr int SCALE_PANEL = Panel, SFA_BYTES = BM * Panel, SFB_BYTES = Panel;
static constexpr int SFA_VECTORS_PER_GROUP = BM / Base::VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + Base::SFA_BYTES_PER_PASS - 1) / Base::SFA_BYTES_PER_PASS;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int OUTPUT_PASSES = BM * Base::B_N / (Base::BLOCK_SIZE * Base::VEC_OUTPUT);
static_assert(BM == 64 || BM == 96 || BM == 128 || BM == 160);
static_assert(Schedule == 1 || Schedule == 2);
static_assert(Schedule != 1 || BM == 128);
static_assert(Stages == 2 || Stages == 3);
static_assert(Panel >= 8 && Panel <= 32 && (Panel & (Panel - 1)) == 0);
static_assert(BM * Base::C_LDS_ROW_STRIDE_ELEMS * 2 <= LDS_BYTES && LDS_BYTES <= 160 * 1024);
};
template<class Base, int Vec, int Block>
struct opus_gemm_mxscale_bpreshuffle_runtime_config_traits : Base {
static constexpr int REDUCE_VEC = Vec, REDUCE_BLOCK = Block;
static constexpr int lds_bytes(int k, int split_k) {
const int total = k / Base::B_K;
const int loops = total / split_k + (total % split_k != 0);
const int stages = loops < Base::NUM_STAGES ? loops : Base::NUM_STAGES;
return stages * (Base::A_STAGE + Base::B_STAGE) +
(Base::REGISTER_SCALES ? 0 : (Base::B_M + Base::B_GROUPS) * loops);
}
};
template<int Panel, int FixedK, bool DirectB, int ChunkRows>
struct opus_gemm_mxscale_bpreshuffle_large_config_traits
: opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950 {
static constexpr int FIXED_K = FixedK, MAX_K = Panel * 128;
static constexpr int SCALE_PANEL = Panel, SFA_BYTES = B_M * Panel;
static constexpr int SFB_BYTES = Panel * sizeof(unsigned);
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / Panel;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES = (B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static constexpr int NUM_STAGES = DirectB ? 3 : 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + (DirectB ? 0 : B_STAGE));
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int B_DIRECT_SETS = 2, C_CHUNK_ROWS = DirectB ? ChunkRows : B_M;
static constexpr int C_CHUNKS = B_M / C_CHUNK_ROWS;
static constexpr int C_CHUNK_BYTES = C_CHUNK_ROWS * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t);
static constexpr int CHUNK_OUTPUT_PASSES = C_CHUNK_ROWS * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static_assert(Panel == 16 || Panel == 32 || Panel == 64 || Panel == 128);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK % 128 == 0 && FixedK <= MAX_K));
static_assert(B_M % C_CHUNK_ROWS == 0 && C_CHUNK_ROWS % 16 == 0);
static_assert(C_CHUNK_ROWS * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(C_CHUNK_BYTES <= LDS_BYTES && LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

struct opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int VEC_SF = 4;
static constexpr int VEC_SF_PAIR = 2 * VEC_SF;
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 : opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int LOOP_UNROLL = 2;
static constexpr bool RESET_SFA_BEFORE_LOAD = false;
static constexpr int B_M=256;
static constexpr int B_N=256;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M / 2;
static constexpr int HALF_B_N = B_N / 2;
static_assert(NUM_WAVES == 4);
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(HALF_B_M % (W_M * T_M) == 0);
static_assert(HALF_B_N % (W_N * T_N) == 0);
static_assert(B_K % (W_K * T_K) == 0);
static constexpr int E_M = HALF_B_M / (W_M * T_M);
static constexpr int E_N = HALF_B_N / (W_N * T_N);
static constexpr int E_K = B_K / (W_K * T_K);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int OUTPUT_TILES_PER_WG = 1;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_SCALE_SF = 4;
static constexpr int VEC_SCALE_SF_PAIR = 8;
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MFMA_SCALE_GROUP_K = 32;
static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_sub = smem_linear_wave / B_K;
static constexpr int smem_m_rep = HALF_B_M / smem_sub;
static constexpr int smem_n_rep = HALF_B_N / smem_sub;
static constexpr int smem_padding = 32;
static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
static_assert(NUM_KGROUPS == 4);
static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int a_ds_read_insts = E_M * W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SCALE_PANEL_K_CAPACITY = 64;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M;
static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_PITCH = B_M;
static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;
static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
static_assert(E_M == 4 && E_N == 4 && E_K == 1);
static_assert(VEC_SCALE_SF == E_M);
static_assert(VEC_SCALE_SF_PAIR == 2 * VEC_SCALE_SF);
static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS == SCALE_PANEL_K_CAPACITY);
static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_scale_reset_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr bool RESET_SFA_BEFORE_LOAD = true;
};

__host__ __device__ inline int opus_mxscale_bpreshuffle_ceil_div(int a, int b) {
return (a + b - 1) / b;
}
template<int BlockM, int BlockN, int WaveM, int WaveN, int Prefetch,
int WaveK = 1, int Output = 0, int BCache = 0,
int FixedK = 0, bool PadN = false, bool ReuseBScale = false>
struct opus_gemm_small_register_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K = WaveK;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * WaveK * 64;
static constexpr int E_M = BlockM / (WaveM * W_M);
static constexpr int E_N = BlockN / (WaveN * W_N);
static constexpr int PREFETCH = Prefetch, WAVE_K = WaveK;
static constexpr int B_CACHE = BCache, OUTPUT = Output;
static constexpr int FIXED_K = FixedK;
static constexpr bool N_TAIL = PadN, REUSE_B_SCALE = ReuseBScale;
static_assert(BlockM % (WaveM * W_M) == 0 && BlockN % (WaveN * W_N) == 0);
static_assert(E_M > 0 && E_M <= 4 && E_N > 0 && Prefetch > 0);
static_assert(BLOCK_SIZE <= 512 && WaveK > 0);
static_assert(Output == 0 || Output == 3 || Output == 4);
static_assert(BCache >= 0 && BCache <= 3);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % 128 == 0));
static_assert(!PadN || Output == 4);
static_assert(!ReuseBScale || 128 % BlockN == 0);
static_assert(Output != 4 || WaveK > 1);
static_assert((WaveK - (Output == 4 ? 0 : 1)) * BlockM * BlockN * sizeof(float) <= 160 * 1024);
};
template<int BlockM, int BlockN, int WaveM, int WaveN,
int Stages, int Cluster = 1, int Output = 2,
bool RegisterScales = false, bool XorLds = false, bool EarlyScaleLoads = false,
bool PrefetchBeforeRead = false, bool ReadOnlyDrain = false,
int SplitK = 1, int ReduceVec = 4, int ReduceBlock = 128,
int StoreCache = 0, int FixedK = 0, bool FineMLoads = false>
struct opus_gemm_small_lds_traits_gfx950 {
static constexpr int B_M = BlockM, B_N = BlockN, B_K=128;
static constexpr int T_M = WaveM, T_N = WaveN, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int BLOCK_SIZE = WaveM * WaveN * 64;
static constexpr int WARP_SIZE = 64, NUM_WAVES = WaveM * WaveN;
static constexpr int E_M = BlockM / (WaveM * 16), E_N = BlockN / (WaveN * 16), E_K = 1;
static constexpr int HALF_B_M = BlockM, HALF_B_N = BlockN;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_SCALE_A = 16;
static constexpr int smem_linear_wave = 1024, smem_padding = XorLds ? 0 : 32;
static constexpr int smem_m_rep = BlockM / 8, smem_n_rep = BlockN / 8;
static constexpr int A_STAGE = BlockM / 8 * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = BlockN / 8 * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = Stages, CLUSTER = Cluster;
static constexpr int OUTPUT = Output;
static constexpr bool REGISTER_SCALES = RegisterScales, XOR_LDS = XorLds;
static constexpr bool EARLY_SCALE_LOADS = EarlyScaleLoads;
static constexpr bool PREFETCH_BEFORE_READ = PrefetchBeforeRead;
static constexpr bool READ_ONLY_DRAIN = ReadOnlyDrain;
static constexpr bool FINE_M_LOADS = FineMLoads;
static constexpr int SPLIT_K = SplitK, REDUCE_VEC = ReduceVec, REDUCE_BLOCK = ReduceBlock;
static constexpr int STORE_CACHE = StoreCache, FIXED_K = FixedK;
static constexpr int MAX_K = 16384, MAX_LOOPS = (MAX_K / B_K + SplitK - 1) / SplitK;
static constexpr int B_GROUPS = (BlockN + 127) / 128;
static constexpr int MAX_SFA_BYTES = BlockM * MAX_LOOPS;
static constexpr int MAX_LDS_BYTES = Stages * (A_STAGE + B_STAGE) +
(RegisterScales ? 0 : (BlockM + B_GROUPS) * MAX_LOOPS);
static constexpr int VMEM_TILE = (BlockM + BlockN) * 128 / (BLOCK_SIZE * 16) +
(RegisterScales ? E_M + B_GROUPS : 0);
static_assert(BlockM % (T_M * 16) == 0 && BlockN % (NUM_WAVES * 16) == 0);
static_assert(FineMLoads || BlockM % (BLOCK_SIZE / 8) == 0);
static_assert(E_M > 0 && E_M <= (FineMLoads ? 8 : 4) && E_N > 0 && MAX_LOOPS >= Stages);
static_assert(SplitK == 1 || SplitK == 2 || SplitK == 4 || SplitK == 8);
static_assert(ReduceVec >= 4 && ReduceVec % 4 == 0 && ReduceVec <= 16);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= MAX_K && FixedK % 128 == 0));
static_assert(Cluster > 0 && Stages % Cluster == 0 && Stages >= 2 * Cluster);
static_assert(MAX_LDS_BYTES <= 160 * 1024 && B_GROUPS * MAX_LOOPS <= BLOCK_SIZE);
static_assert(Output == 1 || Output == 2);
static_assert(!XorLds || (NUM_WAVES == 4 && A_STAGE % 4096 == 0 && B_STAGE % 4096 == 0));
static_assert(Output != 1 || (BlockM * (BlockN + 8) * 2 <= A_STAGE + B_STAGE &&
BlockM * BlockN % (BLOCK_SIZE * 8) == 0));
static constexpr int lds_bytes(int k) {
const int loops = (k / B_K + SplitK - 1) / SplitK;
const int stages = loops < NUM_STAGES ? loops : NUM_STAGES;
return stages * (A_STAGE + B_STAGE) + (RegisterScales ? 0 : (BlockM + B_GROUPS) * loops);
}
};
using opus_gemm_mxscale_bpreshuffle_small_register_16x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 6, 1, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_16x16_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 16, 1, 1, 2, 8>;
using opus_gemm_mxscale_bpreshuffle_small_register_32x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<32, 32, 1, 1, 3, 4, 3, 0, 0, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 1, 4, 8, 2, 2, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_64x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 4, 1, 2, false, false, true, true, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_96x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<96, 64, 2, 2, 4, 1, 1, false, false, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_64x128_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<64, 128, 4, 2, 6, 2>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 2, 2, 4, 1, 2, true>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 2, 2, 4, 1, 2, true, true>;
using opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x128_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 128, 2, 2, 4, 2, 2, true, true, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x16_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 16, 1, 1, 3, 8, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_prefetch_16x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 3, 4, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_16x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<16, 32, 1, 1, 2, 8, 3, 3>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x32_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<32, 32, 1, 1, 2, 8, 3, 3, 0, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_register_wavek_32x64_traits_gfx950 =
opus_gemm_small_register_traits_gfx950<32, 64, 1, 1, 2, 4, 3, 3, 0, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_deep_32x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<32, 64, 1, 4, 12, 4, 2, false, false, true>;
using opus_gemm_mxscale_bpreshuffle_small_lds_deep_64x64_traits_gfx950 =
opus_gemm_small_lds_traits_gfx950<64, 64, 2, 2, 8, 2, 2, false, false, true>;
constexpr int opus_gemm_mxscale_bpreshuffle_small_gcd(int a, int b) { while (b) { int r=a%b; a=b; b=r; } return a; }
template<class Baseline, int Ahead>
struct opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits : Baseline {
static constexpr int B_AHEAD = Ahead, B_SLOTS = Ahead + 1;
static constexpr int RING_PERIOD = Baseline::NUM_STAGES * B_SLOTS / opus_gemm_mxscale_bpreshuffle_small_gcd(Baseline::NUM_STAGES, B_SLOTS);
static constexpr int B_STAGE = 0;
static constexpr int C_BYTES = Baseline::OUTPUT == 1 ? Baseline::B_M * (Baseline::B_N + 8) * 2 : 0;
static constexpr int MAX_LDS_BYTES = Baseline::NUM_STAGES * Baseline::A_STAGE +
(Baseline::REGISTER_SCALES ? 0 : (Baseline::B_M + Baseline::B_GROUPS) * Baseline::MAX_LOOPS);
static_assert(Ahead >= 1 && Ahead <= 3 && Baseline::T_K == 1);
static_assert(MAX_LDS_BYTES <= 160 * 1024 && C_BYTES <= 160 * 1024);
static constexpr int lds_bytes(int k) {
const int loops = (k / Baseline::B_K + Baseline::SPLIT_K - 1) / Baseline::SPLIT_K;
const int stages = loops < Baseline::NUM_STAGES ? loops : Baseline::NUM_STAGES;
const int matrix_scale = stages * Baseline::A_STAGE +
(Baseline::REGISTER_SCALES ? 0 : (Baseline::B_M + Baseline::B_GROUPS) * loops);
return matrix_scale > C_BYTES ? matrix_scale : C_BYTES;
}
};
template<int Actual> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits;
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9043> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9044> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_64x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9045> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_96x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9046> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_64x128_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9047> { using type = opus_gemm_mxscale_bpreshuffle_small_regscale_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9049> { using type = opus_gemm_mxscale_bpreshuffle_small_regscale_xor_32x128_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9055> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_deep_32x64_traits_gfx950; };
template<> struct opus_gemm_mxscale_bpreshuffle_small_baseline_traits<9056> { using type = opus_gemm_mxscale_bpreshuffle_small_lds_deep_64x64_traits_gfx950; };
template<int Actual, int Ahead> using opus_gemm_mxscale_bpreshuffle_small_direct_b_traits = opus_gemm_mxscale_bpreshuffle_small_direct_b_base_traits<typename opus_gemm_mxscale_bpreshuffle_small_baseline_traits<Actual>::type, Ahead>;
template<int BM, int BN, int WaveK>
struct opus_gemm_mxscale_bpreshuffle_register_split_traits
: opus_gemm_small_register_traits_gfx950<BM, BN, 1, 1, 3, WaveK, 3, 3> {
static constexpr int GLOBAL_SPLIT_K = 4;
static_assert(WaveK == 1 || WaveK == 2);
static_assert((BM == 16 && (BN == 16 || BN == 32)) ||
(BM == 32 && (BN == 32 || BN == 64)));
};
template<int BM, int BN, int WM, int WN, int SplitK>
struct opus_gemm_mxscale_bpreshuffle_narrow_fine_traits
: opus_gemm_small_lds_traits_gfx950<BM, BN, WM, WN, 4, 1, 2,
false, false, true, true, true, SplitK, 16, 128, 2, 0, true> {
static_assert((BM == 48 && BN == 64 && WM == 1 && WN == 4) ||
((BM == 64 || BM == 96) && BN == 128 && WM == 2 && WN == 2));
static_assert(SplitK == 1 || SplitK == 2);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=128;
static constexpr int B_N=128;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 128 && B_N == 128 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static_assert(E_M == 4 && E_N == 4 && E_K == 1);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static_assert(MAX_K_TILES == 128);
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 3;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static_assert(NUM_STAGES == 3 && MATRIX_LDS_BYTES == 101376);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;
static_assert(A_VMEM_INSTRUCTIONS == 4 && B_VMEM_INSTRUCTIONS == 4);
static_assert(VMEM_INSTRUCTIONS_PER_TILE == 8);
static constexpr int SCALE_PANEL = 32;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static_assert(SCALE_PANEL == 32 && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(B_M % VEC_SCALE_A == 0 && SFA_BYTES % VEC_SCALE_A == 0);
static_assert(SFA_PASSES == 1 && SFB_BYTES <= BLOCK_SIZE);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(B_SCALE_PACKS == 1);
static_assert(LDS_BYTES == 105504 && LDS_BYTES <= 160 * 1024);
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(C_LDS_ROW_STRIDE_ELEMS == 136);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0>
struct opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=512;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M=4;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert((B_M == 128 || B_M == 192) && (B_N == 128 || B_N == 256));
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr int SWIZZLE_GROUP_M = 4;
static constexpr int SWIZZLE_GROUP_N = 8;
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = 2048;
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = ScalePanel * GROUP_K;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int MATRIX_VMEM_INSTRUCTIONS =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static_assert(NUM_STAGES == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = SCALE_PANEL < 32 ? SCALE_PANEL : 32;
static constexpr int SFA_K_PASSES = SCALE_PANEL / SFA_K_PANEL;
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SFA_K_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(SFA_THREADS_PER_GROUP <= WARP_SIZE);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= MAX_K));
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SFA_K_PANEL);
static_assert(SFA_K_PANEL * SFA_K_PASSES == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS >= B_M);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
using opus_gemm_mxscale_bpreshuffle_8wave_192x256_traits_gfx950 = opus_gemm_mxscale_bpreshuffle_8wave_traits_gfx950<>;
struct opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=160;
static constexpr int B_N=128;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 160 && B_N == 128 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static_assert(E_M == 5 && E_N == 4 && E_K == 1);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static_assert(MAX_K_TILES == 128);
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 76032);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;
static_assert(A_VMEM_INSTRUCTIONS == 5 && B_VMEM_INSTRUCTIONS == 4);
static_assert(VMEM_INSTRUCTIONS_PER_TILE == 9);
static constexpr int SCALE_PANEL = 32;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static_assert(SCALE_PANEL == 32 && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(B_M % VEC_SCALE_A == 0 && SFA_BYTES % VEC_SCALE_A == 0);
static_assert(SFA_PASSES == 2 && SFB_BYTES <= BLOCK_SIZE);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(B_SCALE_PACKS == 1);
static_assert(LDS_BYTES == 81184 && LDS_BYTES <= 80 * 1024);
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(C_LDS_ROW_STRIDE_ELEMS == 136);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
};
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
template<int BM, int BN, int WaveK>
struct opus_gemm_mxscale_bpreshuffle_register_runtime_traits
: opus_gemm_small_register_traits_gfx950<BM, BN, 1, 1, 3, WaveK, 3, 3> {
static constexpr int REDUCE_VEC = 16, REDUCE_BLOCK = 128;
static_assert(WaveK == 1 || WaveK == 2);
static_assert((BM == 16 && (BN == 16 || BN == 32)) ||
(BM == 32 && (BN == 32 || BN == 64)));
};
template<int TileM>
struct opus_gemm_mxscale_bpreshuffle_2wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=128;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 2;
static constexpr int B_M = TileM;
static constexpr int B_N=64;
static constexpr int B_K=128;
static constexpr int T_M=1;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = 32;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static constexpr int B_SCALE_PACKS = (B_N + GROUP_N - 1) / GROUP_N;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(B_M == 16 || B_M == 32);
static_assert(NUM_WAVES == T_M * T_N * T_K && NUM_WAVES == 2);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M % (BLOCK_SIZE / (B_K / VEC_A)) == 0);
static_assert(B_N % (NUM_WAVES * W_N) == 0);
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(E_M <= 2 && E_N == 2 && E_K == 1);
static_assert(NUM_STAGES == 2 && B_K == GROUP_K && B_N <= GROUP_N);
static_assert(SFA_PASSES == 1 && B_SCALE_PACKS == 1);
static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(LDS_BYTES == (B_M == 16 ? 21664 : 26400));
static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * MIN_WGS_PER_CU <= 160 * 1024);
};
using opus_gemm_mxscale_bpreshuffle_2wave_16x64_traits_gfx950 =
opus_gemm_mxscale_bpreshuffle_2wave_traits_gfx950<16>;
struct opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950 {
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192;
static constexpr int B_N=256;
static constexpr int B_K=128;
static constexpr int T_M=4;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 192 && B_N == 256 && B_K == 128);
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int SCALE_N_HALVES = B_N / GROUP_N;
static constexpr int SWIZZLE_GROUP_M = 4;
static constexpr int SWIZZLE_MAX_N = 2048;
static constexpr int SWIZZLE_MIN_M = 4096;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int LOOP_UNROLL = 2;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int smem_padding = 32;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / (B_K / VEC_A);
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = 128;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / SCALE_PANEL;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES =
(SFA_BYTES + BLOCK_SIZE * VEC_SCALE_A - 1) / (BLOCK_SIZE * VEC_SCALE_A);
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = 1;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M % VEC_SCALE_A == 0);
static_assert(BLOCK_SIZE % SCALE_PANEL == 0);
static_assert(WARP_SIZE % SFA_THREADS_PER_GROUP == 0);
static_assert(T_N * T_M * SFA_K_COLUMNS_PER_WAVE == SCALE_PANEL);
static_assert(SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 8 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SCALE_PANEL == 128 && SCALE_PANEL * B_K == MAX_K);
static_assert(SFA_BYTES == 24576 && SFA_PASSES == 3 && SFB_BYTES == 512);
static_assert(B_SCALE_PACKS == 1);
static_assert(LDS_BYTES == 143360 && LDS_BYTES <= 160 * 1024);
};
template<int FixedK>
struct opus_gemm_mxscale_bpreshuffle_shortk_traits
: opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
static constexpr int FIXED_K = FixedK;
static constexpr int SCALE_PANEL = 8;
static constexpr int MAX_K = FIXED_K;
static constexpr int MAX_K_TILES = FIXED_K / B_K;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(FIXED_K == 384 || FIXED_K == 768);
static_assert(MAX_K_TILES == 3 || MAX_K_TILES == 6);
static_assert(MAX_K_TILES <= SCALE_PANEL && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(SFA_BYTES == 1280 && SFB_BYTES == 8 && SFA_PASSES == 1);
static_assert(SFA_VECTORS_PER_GROUP == 10 && B_M % VEC_SCALE_A == 0);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 76032 && LDS_BYTES == 77320);
static_assert(B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr bool PAD_M = true;
};
struct opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_gfx950 {
static constexpr int LOOP_UNROLL = 4;
};
template<int Waves, int ScalePanel = 64>
struct opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950 {
static_assert(Waves == 4 || Waves == 8);
static_assert(ScalePanel == 64 || ScalePanel == 128);
static constexpr int BLOCK_SIZE = Waves * 64;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = Waves;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M = Waves / 2, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % Waves == 0 && smem_n_rep % Waves == 0);
static_assert(E_N == 8 && C_REGS <= 256);
static_assert(LDS_BYTES <= 160 * 1024);
};
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
template<int ScalePanel = 32, int FixedK = 0, int GroupM = 0>
struct opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int FIXED_K = FixedK;
static constexpr int BLOCK_GROUP_M = GroupM;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=64;
static constexpr int B_N=64;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 64 && B_N == 64 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_K_VECTORS = B_K / VEC_A;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / A_K_VECTORS;
static constexpr int A_ROWS_PER_BLOCK = BLOCK_SIZE / A_K_VECTORS;
static constexpr int A_LOAD_PASSES = B_M / A_ROWS_PER_BLOCK;
static constexpr int A_LANES_K = WARP_SIZE / W_M;
static constexpr int A_CHUNK_STRIDE = A_LANES_K * VEC_A;
static_assert(E_M == 2 && E_K == 1);
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(A_K_VECTORS == 8 && A_LANES_K == 4 && A_ROWS_PER_WAVE == W_M / T_M);
static_assert(B_M % A_ROWS_PER_BLOCK == 0);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static_assert(B_N <= GROUP_N && B_K == GROUP_K);
static constexpr int SWIZZLE_PAIR_B_N = 64;
static constexpr unsigned SWIZZLE_MAX_GRID_N = 16;
static constexpr unsigned SWIZZLE_PAIR_MIN_GRID_M = 16;
static constexpr unsigned SWIZZLE_PAIR_MAX_GRID_M = 32;
static constexpr unsigned SWIZZLE_N_PARTITIONS = 4;
static constexpr unsigned SWIZZLE_M_PAIR = 2;
static constexpr unsigned SWIZZLE_PAIR_PARTITIONS = SWIZZLE_N_PARTITIONS * SWIZZLE_M_PAIR;
static constexpr int SWIZZLE_PAIR_SHIFT = 3;
static constexpr unsigned SWIZZLE_MIN_TILES = 256;
static constexpr unsigned SWIZZLE_SMALL_GRID_N = 8;
static constexpr unsigned SWIZZLE_SMALL_TILES = 768;
static constexpr unsigned SWIZZLE_SHORT_GRID_M = 24;
static constexpr unsigned SWIZZLE_LARGE_PARTITIONS = 16;
static_assert(SWIZZLE_PAIR_PARTITIONS == (1u << SWIZZLE_PAIR_SHIFT));
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 4;
static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
static_assert(NUM_STAGES == 4 && VEC_A == VEC_B);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_PRODUCERS_PER_GROUP;
static constexpr int SFA_WORDS = VEC_SCALE_A / sizeof(opus::u32_t);
static constexpr int VEC_SCALE_PACK_A = VEC_SCALE_A / sizeof(opus::u16_t);
static constexpr int SFA_PACK_CHUNKS = VEC_SCALE_A / VEC_SCALE_PACK_A;
static constexpr int SFA_WORDS_PER_CHUNK = SFA_WORDS / SFA_PACK_CHUNKS;
static constexpr unsigned SFA_PACK_PERM_LO = 0x05010400u;
static constexpr unsigned SFA_PACK_PERM_HI = 0x07030602u;
static constexpr int B_SCALE_PACKS = (B_N + GROUP_N - 1) / GROUP_N;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= SCALE_PANEL * B_K));
static_assert(SFA_PRODUCERS_PER_GROUP == 2 && SFA_PACK_CHUNKS == 2 && SFA_WORDS_PER_CHUNK == 2);
static_assert(WARP_SIZE % SFA_PRODUCERS_PER_GROUP == 0);
static_assert(SFA_ROWS_PER_REPEAT * E_M == B_M && E_M == sizeof(opus::u16_t));
static_assert(B_SCALE_PACKS == 1);
static_assert(SCALE_PANEL <= NUM_WAVES * SFA_K_COLUMNS_PER_WAVE && SCALE_PANEL <= BLOCK_SIZE);
static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * 2 <= 160 * 1024);
};
using opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_gfx950 = opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_base_gfx950<>;
template<int BM,int Stages,int Panel,int FixedK=0>
struct opus_gemm_mxscale_bpreshuffle_geometry_traits : opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950 {
static constexpr int B_M=BM, HALF_B_M=BM, FIXED_K=FixedK;
static constexpr int E_M=B_M/(T_M*W_M);
static constexpr int smem_m_rep=B_M*B_K/smem_linear_wave;
static constexpr int A_STAGE=smem_m_rep*(smem_linear_wave+smem_padding);
static constexpr int NUM_STAGES=Stages;
static constexpr int MATRIX_LDS_BYTES=NUM_STAGES*(A_STAGE+B_STAGE);
static constexpr int A_VMEM_INSTRUCTIONS=B_M*B_K/(BLOCK_SIZE*VEC_A);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE=A_VMEM_INSTRUCTIONS+B_VMEM_INSTRUCTIONS;
static constexpr int SCALE_PANEL=Panel;
static constexpr int SFA_BYTES=B_M*SCALE_PANEL;
static constexpr int SFB_BYTES=SCALE_PANEL;
static constexpr int SFA_VECTORS_PER_GROUP=B_M/VEC_SCALE_A;
static constexpr int SFA_PASSES=(SFA_BYTES+SFA_BYTES_PER_PASS-1)/SFA_BYTES_PER_PASS;
static constexpr int A_SCALE_PACKS=(E_M+3)/4;
static constexpr int LDS_BYTES=MATRIX_LDS_BYTES+SFA_BYTES+SFB_BYTES;
static constexpr int OUTPUT_PASSES=B_M*B_N/(BLOCK_SIZE*VEC_OUTPUT);
static_assert(BM==64||BM==96||BM==128||BM==160);
static_assert(Stages==2||Stages==3);
static_assert((Panel&(Panel-1))==0 && Panel>=8 && Panel<=32);
static_assert(FixedK==0 || (FixedK%128==0 && FixedK<=16384));
static_assert(B_M%(T_M*W_M)==0 && smem_m_rep%NUM_WAVES==0);
static_assert(B_M*B_N%(BLOCK_SIZE*VEC_OUTPUT)==0);
static_assert(B_M*C_LDS_ROW_STRIDE_ELEMS*2<=LDS_BYTES && LDS_BYTES<=160*1024);
};
template<int K,int Panel,bool Reset=false>
struct opus_gemm_mxscale_bpreshuffle_pin_fixed_traits : opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr int FIXED_K=K;
static constexpr bool RESET_SFA_BEFORE_LOAD=Reset;
static constexpr int SCALE_PANEL_K_CAPACITY=Panel;
static constexpr int SFA_PASSES_PER_CACHE_PANEL=Panel/SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_BYTES=B_M*Panel;
static constexpr int SFB_PANEL_BYTES=SCALE_N_HALVES*Panel*VEC_SF;
static constexpr int LDS_BYTES=SMEM_A_ELEMS+SMEM_B_ELEMS+SFA_PANEL_BYTES+SFB_PANEL_BYTES;
static_assert(K%128==0 && K<=16384 && Panel>=16 && Panel<=64);
static_assert((Panel&(Panel-1))==0 && SFA_PASSES_PER_CACHE_PANEL>0);
};
template<int K> struct opus_gemm_mxscale_bpreshuffle_pad_pin_fixed_traits : opus_gemm_mxscale_bpreshuffle_4wave_256x256_padded_m_traits_unroll4_gfx950 {
static constexpr int FIXED_K=K;
};
template<int NumStages = 3, int ScalePanel = 32, int FixedK = 0>
struct opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int FIXED_K = FixedK;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=64;
static constexpr int B_N=128;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M;
static constexpr int HALF_B_N = B_N;
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(B_M == 64 && B_N == 128 && B_K == 128);
static_assert(NUM_WAVES == 4 && BLOCK_SIZE == 256 && T_M == 2 && T_N == 2);
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_K_VECTORS = B_K / VEC_A;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / A_K_VECTORS;
static constexpr int A_ROWS_PER_BLOCK = BLOCK_SIZE / A_K_VECTORS;
static constexpr int A_LOAD_PASSES = B_M / A_ROWS_PER_BLOCK;
static constexpr int A_LANES_K = WARP_SIZE / W_M;
static constexpr int A_CHUNK_STRIDE = A_LANES_K * VEC_A;
static_assert(E_M == 2 && E_K == 1);
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(A_K_VECTORS == 8 && A_LANES_K == 4 && A_ROWS_PER_WAVE == W_M / T_M);
static_assert(B_M % A_ROWS_PER_BLOCK == 0);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static_assert(B_N <= GROUP_N && B_K == GROUP_K);
static constexpr int SWIZZLE_PAIR_B_N = 64;
static constexpr unsigned SWIZZLE_MAX_GRID_N = 16;
static constexpr unsigned SWIZZLE_PAIR_MIN_GRID_M = 16;
static constexpr unsigned SWIZZLE_PAIR_MAX_GRID_M = 32;
static constexpr unsigned SWIZZLE_N_PARTITIONS = 4;
static constexpr unsigned SWIZZLE_M_PAIR = 2;
static constexpr unsigned SWIZZLE_PAIR_PARTITIONS = SWIZZLE_N_PARTITIONS * SWIZZLE_M_PAIR;
static constexpr int SWIZZLE_PAIR_SHIFT = 3;
static constexpr unsigned SWIZZLE_MIN_TILES = 256;
static constexpr unsigned SWIZZLE_SMALL_GRID_N = 8;
static constexpr unsigned SWIZZLE_SMALL_TILES = 768;
static constexpr unsigned SWIZZLE_SHORT_GRID_M = 24;
static constexpr unsigned SWIZZLE_LARGE_PARTITIONS = 16;
static_assert(SWIZZLE_PAIR_PARTITIONS == (1u << SWIZZLE_PAIR_SHIFT));
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = NumStages;
static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE =
B_M * B_K / (BLOCK_SIZE * VEC_A) + B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
static_assert(NUM_STAGES >= 2 && NUM_STAGES <= 4 && VEC_A == VEC_B);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_PRODUCERS_PER_GROUP;
static constexpr int SFA_WORDS = VEC_SCALE_A / sizeof(opus::u32_t);
static constexpr int VEC_SCALE_PACK_A = VEC_SCALE_A / sizeof(opus::u16_t);
static constexpr int SFA_PACK_CHUNKS = VEC_SCALE_A / VEC_SCALE_PACK_A;
static constexpr int SFA_WORDS_PER_CHUNK = SFA_WORDS / SFA_PACK_CHUNKS;
static constexpr unsigned SFA_PACK_PERM_LO = 0x05010400u;
static constexpr unsigned SFA_PACK_PERM_HI = 0x07030602u;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert((SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(FIXED_K == 0 || (FIXED_K % B_K == 0 && FIXED_K <= SCALE_PANEL * B_K));
static_assert(SFA_PRODUCERS_PER_GROUP == 2 && SFA_PACK_CHUNKS == 2 && SFA_WORDS_PER_CHUNK == 2);
static_assert(WARP_SIZE % SFA_PRODUCERS_PER_GROUP == 0);
static_assert(SFA_ROWS_PER_REPEAT * E_M == B_M && E_M == sizeof(opus::u16_t));
static_assert(B_SCALE_PACKS == 1);
static_assert(SCALE_PANEL <= NUM_WAVES * SFA_K_COLUMNS_PER_WAVE && SCALE_PANEL <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};
using opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_gfx950 = opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_base_gfx950<>;
using opus_gemm_mxscale_bpreshuffle_2wave_32x64_traits_gfx950 =
opus_gemm_mxscale_bpreshuffle_2wave_traits_gfx950<32>;
template<int BlockM, int WaveM, int WaveN, int Stages, int Cluster, int SplitK,
int ReduceVec=4, int ReduceBlock=128, int StoreCache=0, int FixedK=0>
struct opus_gemm_mxscale_bpreshuffle_fine_traits_gfx950
: opus_gemm_small_lds_traits_gfx950<
BlockM, 128, WaveM, WaveN, Stages, Cluster, 2,
false, false, true, true, true,
SplitK, ReduceVec, ReduceBlock, StoreCache, FixedK, true> {};
namespace opus_bpreshuffle_policy {
template<class Base> constexpr int fixed_k() {
if constexpr (requires { Base::FIXED_K; }) return Base::FIXED_K;
else return 0;
}
template<class Base> constexpr int block_group_m() {
if constexpr (requires { Base::BLOCK_GROUP_M; }) return Base::BLOCK_GROUP_M;
else return 0;
}
}
template<class Base, int Schedule>
struct opus_gemm_mxscale_bpreshuffle_pipeline_traits : Base {
static constexpr int SCHEDULE = Schedule;
static constexpr int FIXED_K = opus_bpreshuffle_policy::fixed_k<Base>();
static constexpr int BLOCK_GROUP_M = opus_bpreshuffle_policy::block_group_m<Base>();
static constexpr int MIN_WGS_PER_CU = 1;
};

template<int Panel, int FixedK, bool Reset, int Unroll>
struct opus_gemm_mxscale_bpreshuffle_pin_config_traits
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr int FIXED_K = FixedK, LOOP_UNROLL = Unroll;
static constexpr bool RESET_SFA_BEFORE_LOAD = Reset;
static constexpr int SCALE_PANEL_K_CAPACITY = Panel;
static constexpr int SFA_PASSES_PER_CACHE_PANEL = Panel / SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_BYTES = B_M * Panel;
static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * Panel * VEC_SF;
static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
static_assert(Panel == 16 || Panel == 32 || Panel == 64);
static_assert(Unroll == 2 || Unroll == 4);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK % 128 == 0 && FixedK <= 16384));
static_assert(LDS_BYTES <= 160 * 1024);
};
template<int BM, int Stages, int Panel, int FixedK, int Schedule>
struct opus_gemm_mxscale_bpreshuffle_small_config_traits
: std::conditional_t<Schedule == 1,
opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950,
opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950> {
using Base = std::conditional_t<Schedule == 1,
opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950,
opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950>;
static constexpr int B_M = BM, HALF_B_M = BM, FIXED_K = FixedK;
static constexpr int E_M = BM / (Base::T_M * Base::W_M);
static constexpr int smem_m_rep = BM * Base::B_K / Base::smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (Base::smem_linear_wave + Base::smem_padding);
static constexpr int NUM_STAGES = Stages;
static constexpr int MATRIX_LDS_BYTES = Stages * (A_STAGE + Base::B_STAGE);
static constexpr int A_VMEM_INSTRUCTIONS = BM * Base::B_K / (Base::BLOCK_SIZE * Base::VEC_A);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + Base::B_VMEM_INSTRUCTIONS;
static constexpr int SCALE_PANEL = Panel, SFA_BYTES = BM * Panel, SFB_BYTES = Panel;
static constexpr int SFA_VECTORS_PER_GROUP = BM / Base::VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + Base::SFA_BYTES_PER_PASS - 1) / Base::SFA_BYTES_PER_PASS;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int OUTPUT_PASSES = BM * Base::B_N / (Base::BLOCK_SIZE * Base::VEC_OUTPUT);
static_assert(BM == 64 || BM == 96 || BM == 128 || BM == 160);
static_assert(Schedule == 1 || Schedule == 2);
static_assert(Schedule != 1 || BM == 128);
static_assert(Stages == 2 || Stages == 3);
static_assert(Panel >= 8 && Panel <= 32 && (Panel & (Panel - 1)) == 0);
static_assert(BM * Base::C_LDS_ROW_STRIDE_ELEMS * 2 <= LDS_BYTES && LDS_BYTES <= 160 * 1024);
};
template<class Base, int Vec, int Block>
struct opus_gemm_mxscale_bpreshuffle_runtime_config_traits : Base {
static constexpr int REDUCE_VEC = Vec, REDUCE_BLOCK = Block;
static constexpr int lds_bytes(int k, int split_k) {
const int total = k / Base::B_K;
const int loops = total / split_k + (total % split_k != 0);
const int stages = loops < Base::NUM_STAGES ? loops : Base::NUM_STAGES;
return stages * (Base::A_STAGE + Base::B_STAGE) +
(Base::REGISTER_SCALES ? 0 : (Base::B_M + Base::B_GROUPS) * loops);
}
};
template<int Panel, int FixedK, bool DirectB, int ChunkRows>
struct opus_gemm_mxscale_bpreshuffle_large_config_traits
: opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950 {
static constexpr int FIXED_K = FixedK, MAX_K = Panel * 128;
static constexpr int SCALE_PANEL = Panel, SFA_BYTES = B_M * Panel;
static constexpr int SFB_BYTES = Panel * sizeof(unsigned);
static constexpr int SFA_THREADS_PER_GROUP = BLOCK_SIZE / Panel;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_THREADS_PER_GROUP;
static constexpr int SFA_ROWS_PER_PASS = SFA_THREADS_PER_GROUP * VEC_SCALE_A;
static constexpr int SFA_PASSES = (B_M + SFA_ROWS_PER_PASS - 1) / SFA_ROWS_PER_PASS;
static constexpr int NUM_STAGES = DirectB ? 3 : 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + (DirectB ? 0 : B_STAGE));
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int B_DIRECT_SETS = 2, C_CHUNK_ROWS = DirectB ? ChunkRows : B_M;
static constexpr int C_CHUNKS = B_M / C_CHUNK_ROWS;
static constexpr int C_CHUNK_BYTES = C_CHUNK_ROWS * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t);
static constexpr int CHUNK_OUTPUT_PASSES = C_CHUNK_ROWS * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static_assert(Panel == 16 || Panel == 32 || Panel == 64 || Panel == 128);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK % 128 == 0 && FixedK <= MAX_K));
static_assert(B_M % C_CHUNK_ROWS == 0 && C_CHUNK_ROWS % 16 == 0);
static_assert(C_CHUNK_ROWS * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
static_assert(C_CHUNK_BYTES <= LDS_BYTES && LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_TILED_265) && !defined(OPUS_BP_INCLUDE_TILED_265_DONE)
#define OPUS_BP_INCLUDE_TILED_265_DONE
namespace opus_bpreshuffle_gfx950::tiled::schedule_265 {
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

struct opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int VEC_SF = 4;
static constexpr int VEC_SF_PAIR = 2 * VEC_SF;
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 : opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int LOOP_UNROLL = 2;
static constexpr bool RESET_SFA_BEFORE_LOAD = false;
static constexpr int B_M=256;
static constexpr int B_N=256;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M / 2;
static constexpr int HALF_B_N = B_N / 2;
static_assert(NUM_WAVES == 4);
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(HALF_B_M % (W_M * T_M) == 0);
static_assert(HALF_B_N % (W_N * T_N) == 0);
static_assert(B_K % (W_K * T_K) == 0);
static constexpr int E_M = HALF_B_M / (W_M * T_M);
static constexpr int E_N = HALF_B_N / (W_N * T_N);
static constexpr int E_K = B_K / (W_K * T_K);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int OUTPUT_TILES_PER_WG = 1;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_SCALE_SF = 4;
static constexpr int VEC_SCALE_SF_PAIR = 8;
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MFMA_SCALE_GROUP_K = 32;
static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_sub = smem_linear_wave / B_K;
static constexpr int smem_m_rep = HALF_B_M / smem_sub;
static constexpr int smem_n_rep = HALF_B_N / smem_sub;
static constexpr int smem_padding = 32;
static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
static_assert(NUM_KGROUPS == 4);
static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int a_ds_read_insts = E_M * W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SCALE_PANEL_K_CAPACITY = 64;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M;
static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_PITCH = B_M;
static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;
static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
static_assert(E_M == 4 && E_N == 4 && E_K == 1);
static_assert(VEC_SCALE_SF == E_M);
static_assert(VEC_SCALE_SF_PAIR == 2 * VEC_SCALE_SF);
static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS == SCALE_PANEL_K_CAPACITY);
static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_scale_reset_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr bool RESET_SFA_BEFORE_LOAD = true;
};

__host__ __device__ inline int opus_mxscale_bpreshuffle_ceil_div(int a, int b) {
return (a + b - 1) / b;
}
#else

// SPDX-License-Identifier: Apache-2.0

struct opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int VEC_SF = 4;
static constexpr int VEC_SF_PAIR = 2 * VEC_SF;
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 : opus_mxscale_bpreshuffle_common_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int LOOP_UNROLL = 2;
static constexpr bool RESET_SFA_BEFORE_LOAD = false;
static constexpr int B_M=256;
static constexpr int B_N=256;
static constexpr int B_K=128;
static constexpr int T_M=2;
static constexpr int T_N=2;
static constexpr int T_K=1;
static constexpr int W_M = 16;
static constexpr int W_N = 16;
static constexpr int W_K = 128;
static constexpr int HALF_B_M = B_M / 2;
static constexpr int HALF_B_N = B_N / 2;
static_assert(NUM_WAVES == 4);
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(HALF_B_M % (W_M * T_M) == 0);
static_assert(HALF_B_N % (W_N * T_N) == 0);
static_assert(B_K % (W_K * T_K) == 0);
static constexpr int E_M = HALF_B_M / (W_M * T_M);
static constexpr int E_N = HALF_B_N / (W_N * T_N);
static constexpr int E_K = B_K / (W_K * T_K);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int OUTPUT_TILES_PER_WG = 1;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_SCALE_SF = 4;
static constexpr int VEC_SCALE_SF_PAIR = 8;
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MFMA_SCALE_GROUP_K = 32;
static constexpr int SCALE_KGROUPS_PER_MFMA = W_K / MFMA_SCALE_GROUP_K;
static constexpr int NUM_KGROUPS = B_K / MFMA_SCALE_GROUP_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_sub = smem_linear_wave / B_K;
static constexpr int smem_m_rep = HALF_B_M / smem_sub;
static constexpr int smem_n_rep = HALF_B_N / smem_sub;
static constexpr int smem_padding = 32;
static constexpr int SCALE_N_HALVES = B_N / HALF_B_N;
static_assert(NUM_KGROUPS == 4);
static constexpr int a_buffer_load_insts = HALF_B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int b_buffer_load_insts = HALF_B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int a_ds_read_insts = E_M * W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int b_ds_read_insts = E_N * W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int SMEM_A_ELEMS = smem_m_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SMEM_B_ELEMS = smem_n_rep * (smem_linear_wave + smem_padding) * 4;
static constexpr int SCALE_PANEL_K_CAPACITY = 64;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / W_M;
static constexpr int SFA_K_COLUMNS_PER_PASS = NUM_WAVES * SFA_K_COLUMNS_PER_WAVE;
static constexpr int SFA_PASSES_PER_CACHE_PANEL = SCALE_PANEL_K_CAPACITY / SFA_K_COLUMNS_PER_PASS;
static constexpr int SFA_PANEL_PITCH = B_M;
static constexpr int SFA_PANEL_BYTES = SFA_PANEL_PITCH * SCALE_PANEL_K_CAPACITY;
static constexpr int SFB_PANEL_BYTES = SCALE_N_HALVES * SCALE_PANEL_K_CAPACITY * VEC_SF;
static constexpr int LDS_BYTES = SMEM_A_ELEMS + SMEM_B_ELEMS + SFA_PANEL_BYTES + SFB_PANEL_BYTES;
static_assert(E_M == 4 && E_N == 4 && E_K == 1);
static_assert(VEC_SCALE_SF == E_M);
static_assert(VEC_SCALE_SF_PAIR == 2 * VEC_SCALE_SF);
static_assert(SFA_PASSES_PER_CACHE_PANEL * SFA_K_COLUMNS_PER_PASS == SCALE_PANEL_K_CAPACITY);
static_assert(a_ds_read_insts == 8 && b_ds_read_insts == 8);
static_assert(SFA_PANEL_BYTES == 16384 && SFB_PANEL_BYTES == 512);
static_assert(LDS_BYTES == 152064 && LDS_BYTES <= 163840);
};
struct opus_gemm_mxscale_bpreshuffle_4wave_traits_scale_reset_gfx950
: opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr bool RESET_SFA_BEFORE_LOAD = true;
};

__host__ __device__ inline int opus_mxscale_bpreshuffle_ceil_div(int a, int b) {
return (a + b - 1) / b;
}

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_TILED_267) && !defined(OPUS_BP_INCLUDE_TILED_267_DONE)
#define OPUS_BP_INCLUDE_TILED_267_DONE
namespace opus_bpreshuffle_gfx950::tiled::schedule_267 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_265;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int Waves, int ScalePanel = 64>
struct opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950 {
static_assert(Waves == 4 || Waves == 8);
static_assert(ScalePanel == 64 || ScalePanel == 128);
static constexpr int BLOCK_SIZE = Waves * 64;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = Waves;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M = Waves / 2, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % Waves == 0 && smem_n_rep % Waves == 0);
static_assert(E_N == 8 && C_REGS <= 256);
static_assert(LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

template<int Waves, int ScalePanel = 64>
struct opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950 {
static_assert(Waves == 4 || Waves == 8);
static_assert(ScalePanel == 64 || ScalePanel == 128);
static constexpr int BLOCK_SIZE = Waves * 64;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = Waves;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M = Waves / 2, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % Waves == 0 && smem_n_rep % Waves == 0);
static_assert(E_N == 8 && C_REGS <= 256);
static_assert(LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_TILED_271) && !defined(OPUS_BP_INCLUDE_TILED_271_DONE)
#define OPUS_BP_INCLUDE_TILED_271_DONE
namespace opus_bpreshuffle_gfx950::tiled::schedule_271 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_267;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

struct opus_gemm_long_runtime_traits
: opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128> {
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int LOOP_UNROLL = 2;
static_assert(B_M == 192 && B_N == 256 && B_K == 128);
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static_assert(E_M == 3 && E_N == 8 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SCALE_PANEL == 128 && SCALE_PANEL * B_K == MAX_K);
static_assert(SFA_BYTES == 24576 && SFA_PASSES == 3 && SFB_BYTES == 256);
static_assert(LDS_BYTES == 143104 && LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

struct opus_gemm_long_runtime_traits
: opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128> {
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int LOOP_UNROLL = 2;
static_assert(B_M == 192 && B_N == 256 && B_K == 128);
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static_assert(E_M == 3 && E_N == 8 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SCALE_PANEL == 128 && SCALE_PANEL * B_K == MAX_K);
static_assert(SFA_BYTES == 24576 && SFA_PASSES == 3 && SFB_BYTES == 256);
static_assert(LDS_BYTES == 143104 && LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_TILED_275) && !defined(OPUS_BP_INCLUDE_TILED_275_DONE)
#define OPUS_BP_INCLUDE_TILED_275_DONE
namespace opus_bpreshuffle_gfx950::tiled::schedule_275 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_267;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

struct opus_gemm_long_runtime_fused_traits
: opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128> {
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int LOOP_UNROLL = 2;
static_assert(B_M == 192 && B_N == 256 && B_K == 128);
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static_assert(E_M == 3 && E_N == 8 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SCALE_PANEL == 128 && SCALE_PANEL * B_K == MAX_K);
static_assert(SFA_BYTES == 24576 && SFA_PASSES == 3 && SFB_BYTES == 256);
static_assert(LDS_BYTES == 143104 && LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

struct opus_gemm_long_runtime_fused_traits
: opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128> {
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int LOOP_UNROLL = 2;
static_assert(B_M == 192 && B_N == 256 && B_K == 128);
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static_assert(E_M == 3 && E_N == 8 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SCALE_PANEL == 128 && SCALE_PANEL * B_K == MAX_K);
static_assert(SFA_BYTES == 24576 && SFA_PASSES == 3 && SFB_BYTES == 256);
static_assert(LDS_BYTES == 143104 && LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_TILED_279) && !defined(OPUS_BP_INCLUDE_TILED_279_DONE)
#define OPUS_BP_INCLUDE_TILED_279_DONE
namespace opus_bpreshuffle_gfx950::tiled::schedule_279 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_267;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

struct opus_gemm_long_runtime_grid_traits
: opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128> {
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int LOOP_UNROLL = 2;
static_assert(B_M == 192 && B_N == 256 && B_K == 128);
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static_assert(E_M == 3 && E_N == 8 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SCALE_PANEL == 128 && SCALE_PANEL * B_K == MAX_K);
static_assert(SFA_BYTES == 24576 && SFA_PASSES == 3 && SFB_BYTES == 256);
static_assert(LDS_BYTES == 143104 && LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

struct opus_gemm_long_runtime_grid_traits
: opus_gemm_mxscale_bpreshuffle_192x256_traits_gfx950<8, 128> {
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int LOOP_UNROLL = 2;
static_assert(B_M == 192 && B_N == 256 && B_K == 128);
static_assert(NUM_WAVES == 8 && BLOCK_SIZE == 512 && T_M == 4 && T_N == 2);
static_assert(E_M == 3 && E_N == 8 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SCALE_PANEL == 128 && SCALE_PANEL * B_K == MAX_K);
static_assert(SFA_BYTES == 24576 && SFA_PASSES == 3 && SFB_BYTES == 256);
static_assert(LDS_BYTES == 143104 && LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_TILED_283) && !defined(OPUS_BP_INCLUDE_TILED_283_DONE)
#define OPUS_BP_INCLUDE_TILED_283_DONE
namespace opus_bpreshuffle_gfx950::tiled::schedule_283 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_265;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

struct runtime_n224_traits {
static constexpr int MAX_K = 1536;
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=224, B_K=128;
static constexpr int T_M=4, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = MAX_K_TILES;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = 3 * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = E_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(NUM_WAVES == 8 && NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 7 && E_K == 1 && C_REGS == 84);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 109824);
static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == 1);
static_assert(SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

struct runtime_n224_traits {
static constexpr int MAX_K = 1536;
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=224, B_K=128;
static constexpr int T_M=4, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = MAX_K_TILES;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = 3 * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = E_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(NUM_WAVES == 8 && NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 7 && E_K == 1 && C_REGS == 84);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 109824);
static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == 1);
static_assert(SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_TILED_287) && !defined(OPUS_BP_INCLUDE_TILED_287_DONE)
#define OPUS_BP_INCLUDE_TILED_287_DONE
namespace opus_bpreshuffle_gfx950::tiled::schedule_287 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_265;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

struct runtime_n224_loop_traits {
static constexpr int MAX_K = 1536;
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=224, B_K=128;
static constexpr int T_M=4, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = MAX_K_TILES;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = 3 * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = E_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(NUM_WAVES == 8 && NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 7 && E_K == 1 && C_REGS == 84);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 109824);
static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == 1);
static_assert(SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

struct runtime_n224_loop_traits {
static constexpr int MAX_K = 1536;
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=224, B_K=128;
static constexpr int T_M=4, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = MAX_K_TILES;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = 3 * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = E_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(NUM_WAVES == 8 && NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 7 && E_K == 1 && C_REGS == 84);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 109824);
static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == 1);
static_assert(SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_TILED_291) && !defined(OPUS_BP_INCLUDE_TILED_291_DONE)
#define OPUS_BP_INCLUDE_TILED_291_DONE
namespace opus_bpreshuffle_gfx950::tiled::schedule_291 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_265;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

struct short_runtime_group4_cache2_traits {
static constexpr int MAX_K = 1536;
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M=4, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = MAX_K_TILES;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(NUM_WAVES == 8 && NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 8 && E_K == 1 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == 1);
static_assert(SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

struct short_runtime_group4_cache2_traits {
static constexpr int MAX_K = 1536;
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M=4, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = MAX_K_TILES;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(NUM_WAVES == 8 && NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 8 && E_K == 1 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == 1);
static_assert(SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_TILED_295) && !defined(OPUS_BP_INCLUDE_TILED_295_DONE)
#define OPUS_BP_INCLUDE_TILED_295_DONE
namespace opus_bpreshuffle_gfx950::tiled::schedule_295 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_265;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

struct short_runtime_unified_traits {
static constexpr int MAX_K = 1536;
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M=4, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = MAX_K_TILES;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(NUM_WAVES == 8 && NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 8 && E_K == 1 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == 1);
static_assert(SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

struct short_runtime_unified_traits {
static constexpr int MAX_K = 1536;
static constexpr int BLOCK_SIZE=512;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int B_M=192, B_N=256, B_K=128;
static constexpr int T_M=4, T_N=2, T_K=1;
static constexpr int W_M = 16, W_N = 16, W_K = 128;
static constexpr int E_M = B_M / (T_M * W_M);
static constexpr int E_N = B_N / (T_N * W_N);
static constexpr int E_K = 1;
static constexpr int VEC_A = 16, VEC_B = 16, VEC_C = 4;
static constexpr int GROUP_M = 1, GROUP_N = 128, GROUP_K = 128;
static constexpr int HALF_B_M = B_M, HALF_B_N = B_N;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = 2;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int SCALE_PANEL = MAX_K_TILES;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = (B_N / GROUP_N) * SCALE_PANEL;
static constexpr int SFA_PASSES = (SFA_BYTES + BLOCK_SIZE * 16 - 1) / (BLOCK_SIZE * 16);
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = B_N / GROUP_N;
static constexpr int C_REGS = B_M * B_N / BLOCK_SIZE;
OPUS_H_D static constexpr int a_lds_offset(int wave_m, int lane, int repeat, int chunk) {
const int matrix_row = repeat * T_M * W_M + wave_m * W_M + lane % W_M;
constexpr int rows_per_pass = BLOCK_SIZE / (B_K / VEC_A);
const int pass = matrix_row / rows_per_pass;
const int remainder = matrix_row % rows_per_pass;
const int producer_n = remainder / (8 * T_M);
const int producer_m = remainder % T_M;
const int producer_lane_row = (remainder % (8 * T_M)) / T_M;
return (pass * NUM_WAVES + producer_n * T_M + producer_m) *
(smem_linear_wave + smem_padding) +
producer_lane_row * B_K + (lane / W_M) * VEC_A + chunk * 64;
}
static_assert(NUM_WAVES == 8 && NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert(E_M == 3 && E_N == 8 && E_K == 1 && C_REGS == 96);
static_assert(NUM_STAGES == 2 && MATRIX_LDS_BYTES == 118272);
static_assert(SFA_BYTES % 16 == 0 && SFA_PASSES == 1);
static_assert(SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif
