#ifndef OPUS_BP_PANEL_TRAITS_BASE
#define OPUS_BP_PANEL_TRAITS_BASE
// SPDX-License-Identifier: Apache-2.0

#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_quadrant_gfx950.cuh"

namespace opus_bpreshuffle_gfx950::panel::schedule_0 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_6;
// SPDX-License-Identifier: Apache-2.0
}

namespace opus_bpreshuffle_gfx950::panel::schedule_1 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int BlockM, int BlockN,
int NumStages = (BlockM == 160 ? 2 : (BlockN == 64 ? 4 : 3)),
int ScalePanel = 32, int FixedK = 0, int GroupM = 0,
bool InterleaveMatrixReads = (BlockN == 128),
bool ByteSafeScaleTail = (BlockM != 160)>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int FIXED_K = FixedK;
static constexpr int BLOCK_GROUP_M = GroupM;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M=2;
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
static constexpr bool PACKED_SCALES = B_M == 64;
static constexpr bool INTERLEAVE_MATRIX_READS = InterleaveMatrixReads;
static constexpr bool BYTE_SAFE_SCALE_TAIL = ByteSafeScaleTail;
static_assert((B_M == 64 && (B_N == 64 || B_N == 128)) ||
((B_M == 128 || B_M == 160) && B_N == 128));
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(BLOCK_GROUP_M >= 0);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_K_VECTORS = B_K / VEC_A;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / A_K_VECTORS;
static constexpr int A_ROWS_PER_BLOCK = BLOCK_SIZE / A_K_VECTORS;
static constexpr int A_LOAD_PASSES = B_M / A_ROWS_PER_BLOCK;
static constexpr int A_LANES_K = WARP_SIZE / W_M;
static constexpr int A_CHUNK_STRIDE = A_LANES_K * VEC_A;
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(A_K_VECTORS == 8 && A_LANES_K == 4 && A_ROWS_PER_WAVE == W_M / T_M);
static_assert(B_M % A_ROWS_PER_BLOCK == 0);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static_assert(B_N <= GROUP_N && B_K == GROUP_K);
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = NumStages;
static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;
static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert(NUM_STAGES >= 2 && NUM_STAGES <= (PACKED_SCALES ? 4 : 3));
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = (B_N + GROUP_N - 1) / GROUP_N;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(SCALE_PANEL > 0 && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(FIXED_K == 0 || (FIXED_K > 0 && FIXED_K <= MAX_K && FIXED_K % B_K == 0));
static_assert(B_SCALE_PACKS == 1 && SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
static_assert(B_N != 64 || ((LDS_BYTES + 1279) / 1280 * 1280) * 2 <= 160 * 1024);
static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_PRODUCERS_PER_GROUP;
static constexpr int SFA_WORDS = VEC_SCALE_A / sizeof(opus::u32_t);
static constexpr int VEC_SCALE_PACK_A = VEC_SCALE_A / sizeof(opus::u16_t);
static constexpr int SFA_PACK_CHUNKS = VEC_SCALE_A / VEC_SCALE_PACK_A;
static constexpr int SFA_WORDS_PER_CHUNK = SFA_WORDS / SFA_PACK_CHUNKS;
static constexpr unsigned SFA_PACK_PERM_LO = 0x05010400u;
static constexpr unsigned SFA_PACK_PERM_HI = 0x07030602u;
static_assert(!PACKED_SCALES || (E_M == sizeof(opus::u16_t) &&
SFA_ROWS_PER_REPEAT * E_M == B_M &&
SCALE_PANEL <= NUM_WAVES * SFA_K_COLUMNS_PER_WAVE));
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M % VEC_SCALE_A == 0 && SFA_BYTES % VEC_SCALE_A == 0);
static_assert(PACKED_SCALES || B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
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
};
}

namespace opus_bpreshuffle_gfx950::panel::schedule_2 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

enum class opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule {
interleaved,
synchronized,
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule Schedule = opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
bool EnableNSwizzle = true, int SwizzleGroupN = 8, int SwizzleGroupM = 4,
int SwizzleMaxN = 2048>
struct opus_gemm_blockscale_mxscale_bpreshuffle_8wave_traits_gfx950 {
static constexpr auto SCHEDULE = Schedule;
static constexpr bool INTERLEAVED = Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved;
static_assert(INTERLEAVED || Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::synchronized);
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
static constexpr bool ENABLE_N_SWIZZLE = EnableNSwizzle;
static constexpr int SWIZZLE_GROUP_M = SwizzleGroupM;
static constexpr int SWIZZLE_GROUP_N = SwizzleGroupN;
static_assert(SWIZZLE_GROUP_M > 0 && SWIZZLE_GROUP_N > 0);
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = SwizzleMaxN;
static_assert(SWIZZLE_MAX_N > 0);
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
static_assert(NUM_STAGES == 2 && LOOP_UNROLL == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = INTERLEAVED && SCALE_PANEL > 32 ? 32 : SCALE_PANEL;
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
static_assert(INTERLEAVED || (B_M == 192 && B_N == 256 && SCALE_PANEL == 128));
static_assert(INTERLEAVED || SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(INTERLEAVED || (smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0));
static_assert(LDS_BYTES <= 160 * 1024);
};
}

namespace opus_bpreshuffle_gfx950::panel::schedule_3 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_1;
using namespace opus_bpreshuffle_gfx950::panel::schedule_2;
// SPDX-License-Identifier: Apache-2.0





namespace opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_layout {
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::A_ROWS_PER_WAVE>{},
opus::number<T::E_K>{},
opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
const int matrix_row = wave_id_m * T::W_M + lane_id_m;
const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
const int producer_m = matrix_row % T::T_M;
const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_THREADS_PER_GROUP, lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id, int stride_sfb) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::NUM_WAVES>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
__device__ inline constexpr auto make_layout_rsfb_scale() {
constexpr auto block_shape = opus::make_tuple(1_I);
constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
}
}

namespace opus_bpreshuffle_gfx950::panel::schedule_4 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_1;
using namespace opus_bpreshuffle_gfx950::panel::schedule_3;
// SPDX-License-Identifier: Apache-2.0


namespace opus_gemm_blockscale_mxscale_bpreshuffle_4wave_128x128_layout {
using opus::operator""_I;
template<int Vec, int Elem, class V, class Smem, class Layout, int Repeat>
__device__ inline void load_matrix_fragment(V& dst, Smem& src, const Layout& layout, opus::number<Repeat> repeat, int offset) {
const auto fragment_layout = opus::make_layout<Vec>(
layout.shape(), layout.stride(),
opus::concat_tuple(opus::make_tuple(repeat), opus::slice(layout.coord(), 1_I, opus::number<Layout::rank>{})));
opus::set_slice(dst, opus::load<Vec>(src, fragment_layout + offset), opus::number<Repeat * Elem>{}, opus::number<(Repeat + 1) * Elem>{});
}
template<class T, class Gmem>
__device__ inline auto load_sfa_vector(Gmem& g_sfa, int gmem_offset, int valid_rows) {
opus::vector_t<unsigned char, T::VEC_SCALE_A> raw;
if (valid_rows >= T::VEC_SCALE_A && gmem_offset % T::VEC_SCALE_A == 0) {
raw = opus::load<T::VEC_SCALE_A>(g_sfa, gmem_offset);
} else {
opus::static_for<T::VEC_SCALE_A>([&](auto byte_i) {
constexpr int byte = decltype(byte_i)::value;
raw[byte] = 0x7f;
if (byte < valid_rows)
raw[byte] = opus::load<1>(g_sfa, gmem_offset + byte)[0];
});
}
return raw;
}
template<class T, int Pass>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
static_assert(Pass >= 0 && Pass < T::SFA_PASSES);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_PANEL>{},
opus::number<T::SFA_VECTORS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int wave_vector_begin = (Pass * T::NUM_WAVES + wave_id_n * T::T_M + wave_id_m) * T::WARP_SIZE;
const int lane_vector = wave_vector_begin % T::SFA_VECTORS_PER_GROUP + lane_id;
const int local_k_group = wave_vector_begin / T::SFA_VECTORS_PER_GROUP + lane_vector / T::SFA_VECTORS_PER_GROUP;
const int row_vector = lane_vector % T::SFA_VECTORS_PER_GROUP;
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{local_k_group, row_vector}));
}
template<class T, int Pass>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
static_assert(Pass >= 0 && Pass < T::SFA_PASSES);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_PANEL>{},
opus::number<T::SFA_VECTORS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int wave_vector_begin = (Pass * T::NUM_WAVES + wave_id_n * T::T_M + wave_id_m) * T::WARP_SIZE;
const int lane_vector = wave_vector_begin % T::SFA_VECTORS_PER_GROUP + lane_id;
const int local_k_group = wave_vector_begin / T::SFA_VECTORS_PER_GROUP + lane_vector / T::SFA_VECTORS_PER_GROUP;
const int row_vector = lane_vector % T::SFA_VECTORS_PER_GROUP;
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{local_k_group, row_vector}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfb) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::B_SCALE_PACKS>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::B_SCALE_PACKS>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::WARP_SIZE>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SCALE_PANEL>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfb_scale() {
constexpr auto block_shape = opus::make_tuple(opus::number<T::B_SCALE_PACKS>{}, 1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SCALE_PANEL>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
}
namespace opus_gemm_blockscale_mxscale_bpreshuffle_4wave_64x128_layout {
using opus::operator""_I;
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::SFA_PRODUCERS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{lane_id % T::SFA_PRODUCERS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PRODUCERS_PER_GROUP>{},
opus::number<T::SFA_PACK_CHUNKS>{},
opus::number<T::VEC_SCALE_PACK_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_PACK_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{lane_id % T::SFA_PRODUCERS_PER_GROUP}));
}
template<class T>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_M>{},
opus::number<T::W_M>{},
1_I);
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
}
namespace opus_gemm_blockscale_mxscale_bpreshuffle_4wave_128x128_layout {
template<int Vec, int Passes, class MakeLayout>
__device__ inline auto make_panel_offsets(MakeLayout make_layout) {
opus::array<int, Passes> offsets;
opus::static_for<Passes>([&](auto pass_i) {
offsets[decltype(pass_i)::value] = opus::layout_to_offsets<Vec>(make_layout(pass_i))[0];
});
return offsets;
}
}
}

namespace opus_bpreshuffle_gfx950::panel::schedule_5 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
// SPDX-License-Identifier: Apache-2.0

template<int TileM, int ScalePanel = 32, int FixedK = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_2wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=128;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 2;
static constexpr int B_M = TileM;
static constexpr int FIXED_K = FixedK;
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
static constexpr int SCALE_PANEL = ScalePanel;
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
static_assert(SCALE_PANEL > 0);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % B_K == 0));
static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * MIN_WGS_PER_CU <= 160 * 1024);
};
}

namespace opus_bpreshuffle_gfx950::panel::schedule_6 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_4;
using namespace opus_bpreshuffle_gfx950::panel::schedule_5;
// SPDX-License-Identifier: Apache-2.0


#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class Traits>
__device__ inline void opus_gemm_blockscale_mxscale_bpreshuffle_2wave(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
namespace layout_9020 = opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_layout;
namespace layout_9021 = opus_gemm_blockscale_mxscale_bpreshuffle_4wave_128x128_layout;
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const int row = block_id_y() * T::B_M;
const int col = block_id_x() * T::B_N;
const int batch_id = block_id_z();
const int loops = (T::FIXED_K ? T::FIXED_K : kargs.k) / T::B_K;
auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a,
static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col,
static_cast<unsigned>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C)));
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = layout_9020::make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = layout_9021::make_layout_gsfa_scale<T, 0>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_ssfa = layout_9021::make_layout_ssfa_scale<T, 0>(lane_id, wave_id_m, wave_id_n);
const auto u_rsfa = layout_9020::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_gsfb = layout_9021::make_layout_gsfb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfb);
const auto u_ssfb = layout_9021::make_layout_ssfb_scale<T>(lane_id, wave_id_m, wave_id_n);
const auto u_rsfb = layout_9021::make_layout_rsfb_scale<T>();
alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
constexpr int ELEM_A = decltype(mma)::elem_a;
constexpr int ELEM_B = decltype(mma)::elem_b;
constexpr int ELEM_C = decltype(mma)::elem_c;
typename decltype(mma)::vtype_a v_a;
typename decltype(mma)::vtype_b v_b;
typename decltype(mma)::vtype_c v_c;
clear(v_c);
const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_ssfa);
const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);
D_SF_PACK v_sfa;
D_SF_PACK v_sfb;
auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int tile_k) { return (tile_k & (T::SCALE_PANEL - 1)) * T::B_M; };
auto ssfb_offset = [&](int tile_k) { return tile_k & (T::SCALE_PANEL - 1); };
auto issue_matrix_prefetch = [&](int stage, int tile_k) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
};
auto load_scale_panel = [&](int panel_k_begin) {
const int sfa_offset = sfa_smem_offsets[0];
const int sfa_group = sfa_offset / T::B_M;
const int sfa_row = sfa_offset % T::B_M;
if (sfa_offset < T::SFA_BYTES && panel_k_begin + sfa_group < loops) {
const auto raw = layout_9021::load_sfa_vector<T>(
g_sfa, sfa_gmem_offsets[0] + gsfa_offset(panel_k_begin),
kargs.m - row - sfa_row);
store<T::VEC_SCALE_A>(s_sfa, raw, sfa_offset);
}
const int sfb_group = sfb_smem_offsets[0];
if (sfb_group < T::SCALE_PANEL && panel_k_begin + sfb_group < loops) {
store<1>(s_sfb, load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin)), sfb_group);
}
};
auto read_scales = [&](int tile_k) {
v_sfa = 0;
static_for<T::E_M>([&](auto m_i) {
constexpr int m_repeat = decltype(m_i)::value;
const D_SF_PACK value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
v_sfa |= value << (m_repeat * 8);
});
v_sfb = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
};
auto load_a_fragment = [&](auto m_i, int stage) {
constexpr int m_repeat = decltype(m_i)::value;
const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(stage));
static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
number<index * T::VEC_A>{}, number<(index + 1) * T::VEC_A>{});
});
};
auto load_b_fragment = [&](auto n_i, int stage) {
constexpr int n_repeat = decltype(n_i)::value;
const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage));
static_for<T::B_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
constexpr int index = n_repeat * T::B_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
set_slice(v_b, load<T::VEC_B>(s_b, offsets[index]),
number<index * T::VEC_B>{}, number<(index + 1) * T::VEC_B>{});
});
};
auto mma_scale_fragment = [&](auto m_i, auto n_i) {
constexpr int m_repeat = decltype(m_i)::value;
constexpr int n_repeat = decltype(n_i)::value;
constexpr int c_index = m_repeat * T::E_N + n_repeat;
const auto a = slice(v_a, number<m_repeat * ELEM_A>{}, number<(m_repeat + 1) * ELEM_A>{});
const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
c = typename decltype(mma)::MMA{}(a, b, c,
static_cast<int>(v_sfa), static_cast<int>(v_sfb),
number<m_repeat>{}, number<0>{});
set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
};
issue_matrix_prefetch(0, 0);
load_scale_panel(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
read_scales(0);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, 0); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, 0); });
s_waitcnt_lgkmcnt(0_I);
int stage = 0;
for (int tile_k = 0; tile_k + 1 < loops; ++tile_k) {
const int next_stage = stage ^ 1;
issue_matrix_prefetch(next_stage, tile_k + 1);
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
});
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (((tile_k + 1) & (T::SCALE_PANEL - 1)) == 0) {
load_scale_panel(tile_k + 1);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
read_scales(tile_k + 1);
static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, next_stage); });
static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, next_stage); });
s_waitcnt_lgkmcnt(0_I);
stage = next_stage;
}
static_for<T::E_M>([&](auto m_i) {
static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
});
static_for<T::E_M * T::E_N>([&](auto c_i) {
constexpr int index = decltype(c_i)::value;
const auto c = slice(v_c, number<index * ELEM_C>{}, number<(index + 1) * ELEM_C>{});
store<T::VEC_C>(g_c, cast<D_C>(c), gc_offsets[index]);
});
}
#endif
}

namespace opus_bpreshuffle_gfx950::panel::schedule_7 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_1;
// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0, bool AScaleDma = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_supply_traits_gfx950 : BaseTraits {
using BASE_TRAITS = BaseTraits;
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static_assert((StaticN == 0) == (StaticStrideK == 0));
static_assert(StaticN == 0 || (StaticN > 0 && StaticN % 128 == 0 && StaticN % BaseTraits::B_N == 0));
static_assert(StaticStrideK == 0 || (StaticStrideK > 0 && StaticStrideK <= BaseTraits::MAX_K &&
StaticStrideK % BaseTraits::B_K == 0));
static_assert(StaticStrideK == 0 || BaseTraits::FIXED_K == 0 || StaticStrideK == BaseTraits::FIXED_K);
static_assert(!AScaleDma || !BaseTraits::PACKED_SCALES,
"Raw SFA DMA requires the LDS four-wave kernel");
};
}

namespace opus_bpreshuffle_gfx950::panel::schedule_8 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_4;
using namespace opus_bpreshuffle_gfx950::panel::schedule_7;
// SPDX-License-Identifier: Apache-2.0
}

namespace opus_bpreshuffle_gfx950::panel::schedule_9 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_2;
// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0,
bool AScaleDma = false, bool SfaWaveorder = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_panel8_supply_traits_gfx950 : BaseTraits {
using BASE_TRAITS = BaseTraits;
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static constexpr bool SFA_WAVEORDER = SfaWaveorder;
static_assert(StaticN == 0 && StaticStrideK == 0,
"Panel supply supports SFA DMA or wave ordering");
static_assert(!AScaleDma || !SfaWaveorder,
"SFA DMA and wave ordering are separate supply configurations");
static_assert(!SfaWaveorder || BaseTraits::INTERLEAVED);
};
}

namespace opus_bpreshuffle_gfx950::panel::schedule_10 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_9;
using namespace opus_bpreshuffle_gfx950::panel::schedule_3;
// SPDX-License-Identifier: Apache-2.0


namespace opus_gemm_blockscale_mxscale_bpreshuffle_panel8_supply_layout {
template<class T, bool WaveStrided = false>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<WaveStrided ? T::SFA_K_COLUMNS_PER_WAVE : T::T_N>{},
opus::number<WaveStrided ? T::T_N : T::T_M>{},
opus::number<WaveStrided ? T::T_M : T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{WaveStrided ? lane_id / T::SFA_THREADS_PER_GROUP : wave_id_n,
WaveStrided ? wave_id_n : wave_id_m,
WaveStrided ? wave_id_m : lane_id / T::SFA_THREADS_PER_GROUP,
lane_id % T::SFA_THREADS_PER_GROUP}));
}
template<class T, bool WaveStrided = false>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<WaveStrided ? T::SFA_K_COLUMNS_PER_WAVE : T::T_N>{},
opus::number<WaveStrided ? T::T_N : T::T_M>{},
opus::number<WaveStrided ? T::T_M : T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::SFA_PASSES>{},
opus::number<T::SFA_THREADS_PER_GROUP>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{WaveStrided ? lane_id / T::SFA_THREADS_PER_GROUP : wave_id_n,
WaveStrided ? wave_id_n : wave_id_m,
WaveStrided ? wave_id_m : lane_id / T::SFA_THREADS_PER_GROUP,
lane_id % T::SFA_THREADS_PER_GROUP}));
}
}
}

#endif

#if defined(OPUS_BP_INCLUDE_PANEL_11) && !defined(OPUS_BP_INCLUDE_PANEL_11_DONE)
#define OPUS_BP_INCLUDE_PANEL_11_DONE
namespace opus_bpreshuffle_gfx950::panel::schedule_11 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int BlockM, int BlockN,
int NumStages = (BlockM == 160 ? 2 : (BlockN == 64 ? 4 : 3)),
int ScalePanel = 32, int FixedK = 0, int GroupM = 0,
bool InterleaveMatrixReads = (BlockN == 128),
bool ByteSafeScaleTail = (BlockM != 160)>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int FIXED_K = FixedK;
static constexpr int BLOCK_GROUP_M = GroupM;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M=2;
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
static constexpr bool PACKED_SCALES = B_M == 64;
static constexpr bool INTERLEAVE_MATRIX_READS = InterleaveMatrixReads;
static constexpr bool BYTE_SAFE_SCALE_TAIL = ByteSafeScaleTail;
static_assert((B_M == 64 && (B_N == 64 || B_N == 128)) ||
((B_M == 128 || B_M == 160) && B_N == 128));
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(BLOCK_GROUP_M >= 0);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_K_VECTORS = B_K / VEC_A;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / A_K_VECTORS;
static constexpr int A_ROWS_PER_BLOCK = BLOCK_SIZE / A_K_VECTORS;
static constexpr int A_LOAD_PASSES = B_M / A_ROWS_PER_BLOCK;
static constexpr int A_LANES_K = WARP_SIZE / W_M;
static constexpr int A_CHUNK_STRIDE = A_LANES_K * VEC_A;
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(A_K_VECTORS == 8 && A_LANES_K == 4 && A_ROWS_PER_WAVE == W_M / T_M);
static_assert(B_M % A_ROWS_PER_BLOCK == 0);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static_assert(B_N <= GROUP_N && B_K == GROUP_K);
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = NumStages;
static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;
static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert(NUM_STAGES >= 2 && NUM_STAGES <= (PACKED_SCALES ? 4 : 3));
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = (B_N + GROUP_N - 1) / GROUP_N;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(SCALE_PANEL > 0 && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(FIXED_K == 0 || (FIXED_K > 0 && FIXED_K <= MAX_K && FIXED_K % B_K == 0));
static_assert(B_SCALE_PACKS == 1 && SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
static_assert(B_N != 64 || ((LDS_BYTES + 1279) / 1280 * 1280) * 2 <= 160 * 1024);
static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_PRODUCERS_PER_GROUP;
static constexpr int SFA_WORDS = VEC_SCALE_A / sizeof(opus::u32_t);
static constexpr int VEC_SCALE_PACK_A = VEC_SCALE_A / sizeof(opus::u16_t);
static constexpr int SFA_PACK_CHUNKS = VEC_SCALE_A / VEC_SCALE_PACK_A;
static constexpr int SFA_WORDS_PER_CHUNK = SFA_WORDS / SFA_PACK_CHUNKS;
static constexpr unsigned SFA_PACK_PERM_LO = 0x05010400u;
static constexpr unsigned SFA_PACK_PERM_HI = 0x07030602u;
static_assert(!PACKED_SCALES || (E_M == sizeof(opus::u16_t) &&
SFA_ROWS_PER_REPEAT * E_M == B_M &&
SCALE_PANEL <= NUM_WAVES * SFA_K_COLUMNS_PER_WAVE));
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M % VEC_SCALE_A == 0 && SFA_BYTES % VEC_SCALE_A == 0);
static_assert(PACKED_SCALES || B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
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
};
#else

// SPDX-License-Identifier: Apache-2.0

template<int BlockM, int BlockN,
int NumStages = (BlockM == 160 ? 2 : (BlockN == 64 ? 4 : 3)),
int ScalePanel = 32, int FixedK = 0, int GroupM = 0,
bool InterleaveMatrixReads = (BlockN == 128),
bool ByteSafeScaleTail = (BlockM != 160)>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=256;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 1;
static constexpr int FIXED_K = FixedK;
static constexpr int BLOCK_GROUP_M = GroupM;
static constexpr int B_M = BlockM;
static constexpr int B_N = BlockN;
static constexpr int B_K=128;
static constexpr int T_M=2;
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
static constexpr bool PACKED_SCALES = B_M == 64;
static constexpr bool INTERLEAVE_MATRIX_READS = InterleaveMatrixReads;
static constexpr bool BYTE_SAFE_SCALE_TAIL = ByteSafeScaleTail;
static_assert((B_M == 64 && (B_N == 64 || B_N == 128)) ||
((B_M == 128 || B_M == 160) && B_N == 128));
static_assert(NUM_WAVES == T_M * T_N * T_K);
static_assert(B_M % (T_M * W_M) == 0 && B_N % (T_N * W_N) == 0);
static_assert(BLOCK_GROUP_M >= 0);
static constexpr int VEC_A = 16;
static constexpr int VEC_B = 16;
static constexpr int VEC_C = 4;
static constexpr int VEC_SCALE_A = 16;
static constexpr int VEC_OUTPUT = 8;
static constexpr int A_CHUNKS_PER_FRAGMENT = W_M * W_K / (WARP_SIZE * VEC_A);
static constexpr int B_CHUNKS_PER_FRAGMENT = W_N * W_K / (WARP_SIZE * VEC_B);
static constexpr int A_K_VECTORS = B_K / VEC_A;
static constexpr int A_ROWS_PER_WAVE = WARP_SIZE / A_K_VECTORS;
static constexpr int A_ROWS_PER_BLOCK = BLOCK_SIZE / A_K_VECTORS;
static constexpr int A_LOAD_PASSES = B_M / A_ROWS_PER_BLOCK;
static constexpr int A_LANES_K = WARP_SIZE / W_M;
static constexpr int A_CHUNK_STRIDE = A_LANES_K * VEC_A;
static_assert(A_CHUNKS_PER_FRAGMENT == 2 && B_CHUNKS_PER_FRAGMENT == 2);
static_assert(A_K_VECTORS == 8 && A_LANES_K == 4 && A_ROWS_PER_WAVE == W_M / T_M);
static_assert(B_M % A_ROWS_PER_BLOCK == 0);
static constexpr int GROUP_M = 1;
static constexpr int GROUP_N = 128;
static constexpr int GROUP_K = 128;
static constexpr int MIN_K = 128;
static constexpr int MAX_K = 16384;
static constexpr int MAX_K_TILES = MAX_K / B_K;
static_assert(B_N <= GROUP_N && B_K == GROUP_K);
static constexpr int smem_linear_wave = WARP_SIZE * VEC_A;
static constexpr int smem_padding = 32;
static constexpr int smem_m_rep = B_M * B_K / smem_linear_wave;
static constexpr int smem_n_rep = B_N * B_K / smem_linear_wave;
static constexpr int A_STAGE = smem_m_rep * (smem_linear_wave + smem_padding);
static constexpr int B_STAGE = smem_n_rep * (smem_linear_wave + smem_padding);
static constexpr int NUM_STAGES = NumStages;
static constexpr int PREFETCH_DISTANCE = NUM_STAGES - 1;
static constexpr int MATRIX_LDS_BYTES = NUM_STAGES * (A_STAGE + B_STAGE);
static constexpr int A_VMEM_INSTRUCTIONS = B_M * B_K / (BLOCK_SIZE * VEC_A);
static constexpr int B_VMEM_INSTRUCTIONS = B_N * B_K / (BLOCK_SIZE * VEC_B);
static constexpr int VMEM_INSTRUCTIONS_PER_TILE = A_VMEM_INSTRUCTIONS + B_VMEM_INSTRUCTIONS;
static constexpr int VMEM_STEADY_WAIT = (PREFETCH_DISTANCE - 1) * VMEM_INSTRUCTIONS_PER_TILE;
static_assert(smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0);
static_assert(NUM_STAGES >= 2 && NUM_STAGES <= (PACKED_SCALES ? 4 : 3));
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int A_SCALE_PACKS = (E_M + 3) / 4;
static constexpr int B_SCALE_PACKS = (B_N + GROUP_N - 1) / GROUP_N;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = B_SCALE_PACKS * SCALE_PANEL;
static constexpr int LDS_BYTES = MATRIX_LDS_BYTES + SFA_BYTES + SFB_BYTES;
static_assert(SCALE_PANEL > 0 && (SCALE_PANEL & (SCALE_PANEL - 1)) == 0);
static_assert(FIXED_K == 0 || (FIXED_K > 0 && FIXED_K <= MAX_K && FIXED_K % B_K == 0));
static_assert(B_SCALE_PACKS == 1 && SFB_BYTES <= BLOCK_SIZE);
static_assert(LDS_BYTES <= 160 * 1024);
static_assert(B_N != 64 || ((LDS_BYTES + 1279) / 1280 * 1280) * 2 <= 160 * 1024);
static constexpr int SFA_ROWS_PER_REPEAT = T_M * W_M;
static constexpr int SFA_PRODUCERS_PER_GROUP = SFA_ROWS_PER_REPEAT / VEC_SCALE_A;
static constexpr int SFA_K_COLUMNS_PER_WAVE = WARP_SIZE / SFA_PRODUCERS_PER_GROUP;
static constexpr int SFA_WORDS = VEC_SCALE_A / sizeof(opus::u32_t);
static constexpr int VEC_SCALE_PACK_A = VEC_SCALE_A / sizeof(opus::u16_t);
static constexpr int SFA_PACK_CHUNKS = VEC_SCALE_A / VEC_SCALE_PACK_A;
static constexpr int SFA_WORDS_PER_CHUNK = SFA_WORDS / SFA_PACK_CHUNKS;
static constexpr unsigned SFA_PACK_PERM_LO = 0x05010400u;
static constexpr unsigned SFA_PACK_PERM_HI = 0x07030602u;
static_assert(!PACKED_SCALES || (E_M == sizeof(opus::u16_t) &&
SFA_ROWS_PER_REPEAT * E_M == B_M &&
SCALE_PANEL <= NUM_WAVES * SFA_K_COLUMNS_PER_WAVE));
static constexpr int SFA_BYTES_PER_PASS = BLOCK_SIZE * VEC_SCALE_A;
static constexpr int SFA_VECTORS_PER_GROUP = B_M / VEC_SCALE_A;
static constexpr int SFA_PASSES = (SFA_BYTES + SFA_BYTES_PER_PASS - 1) / SFA_BYTES_PER_PASS;
static constexpr int C_LDS_ROW_STRIDE_ELEMS = B_N + 8;
static constexpr int OUTPUT_PASSES = B_M * B_N / (BLOCK_SIZE * VEC_OUTPUT);
static_assert(B_M % VEC_SCALE_A == 0 && SFA_BYTES % VEC_SCALE_A == 0);
static_assert(PACKED_SCALES || B_M * C_LDS_ROW_STRIDE_ELEMS * sizeof(opus::bf16_t) <= LDS_BYTES);
static_assert(B_M * B_N % (BLOCK_SIZE * VEC_OUTPUT) == 0);
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
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PANEL_12) && !defined(OPUS_BP_INCLUDE_PANEL_12_DONE)
#define OPUS_BP_INCLUDE_PANEL_12_DONE
namespace opus_bpreshuffle_gfx950::panel::schedule_12 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

enum class opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule {
interleaved,
synchronized,
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule Schedule = opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
bool EnableNSwizzle = true, int SwizzleGroupN = 8, int SwizzleGroupM = 4,
int SwizzleMaxN = 2048>
struct opus_gemm_blockscale_mxscale_bpreshuffle_8wave_traits_gfx950 {
static constexpr auto SCHEDULE = Schedule;
static constexpr bool INTERLEAVED = Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved;
static_assert(INTERLEAVED || Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::synchronized);
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
static constexpr bool ENABLE_N_SWIZZLE = EnableNSwizzle;
static constexpr int SWIZZLE_GROUP_M = SwizzleGroupM;
static constexpr int SWIZZLE_GROUP_N = SwizzleGroupN;
static_assert(SWIZZLE_GROUP_M > 0 && SWIZZLE_GROUP_N > 0);
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = SwizzleMaxN;
static_assert(SWIZZLE_MAX_N > 0);
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
static_assert(NUM_STAGES == 2 && LOOP_UNROLL == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = INTERLEAVED && SCALE_PANEL > 32 ? 32 : SCALE_PANEL;
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
static_assert(INTERLEAVED || (B_M == 192 && B_N == 256 && SCALE_PANEL == 128));
static_assert(INTERLEAVED || SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(INTERLEAVED || (smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0));
static_assert(LDS_BYTES <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

enum class opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule {
interleaved,
synchronized,
};
template<int BlockM = 192, int BlockN = 256, int ScalePanel = 128, int FixedK = 0,
opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule Schedule = opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved,
bool EnableNSwizzle = true, int SwizzleGroupN = 8, int SwizzleGroupM = 4,
int SwizzleMaxN = 2048>
struct opus_gemm_blockscale_mxscale_bpreshuffle_8wave_traits_gfx950 {
static constexpr auto SCHEDULE = Schedule;
static constexpr bool INTERLEAVED = Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::interleaved;
static_assert(INTERLEAVED || Schedule == opus_gemm_blockscale_mxscale_bpreshuffle_8wave_panel_schedule::synchronized);
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
static constexpr bool ENABLE_N_SWIZZLE = EnableNSwizzle;
static constexpr int SWIZZLE_GROUP_M = SwizzleGroupM;
static constexpr int SWIZZLE_GROUP_N = SwizzleGroupN;
static_assert(SWIZZLE_GROUP_M > 0 && SWIZZLE_GROUP_N > 0);
static constexpr int SWIZZLE_MIN_N_TILES = 256;
static constexpr int SWIZZLE_MAX_N = SwizzleMaxN;
static_assert(SWIZZLE_MAX_N > 0);
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
static_assert(NUM_STAGES == 2 && LOOP_UNROLL == 2);
static constexpr int SCALE_PANEL = ScalePanel;
static constexpr int SFA_BYTES = B_M * SCALE_PANEL;
static constexpr int SFB_BYTES = SCALE_PANEL * sizeof(unsigned);
static constexpr int SFA_K_PANEL = INTERLEAVED && SCALE_PANEL > 32 ? 32 : SCALE_PANEL;
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
static_assert(INTERLEAVED || (B_M == 192 && B_N == 256 && SCALE_PANEL == 128));
static_assert(INTERLEAVED || SFA_PASSES * SFA_ROWS_PER_PASS == B_M);
static_assert(INTERLEAVED || (smem_m_rep % NUM_WAVES == 0 && smem_n_rep % NUM_WAVES == 0));
static_assert(LDS_BYTES <= 160 * 1024);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PANEL_16) && !defined(OPUS_BP_INCLUDE_PANEL_16_DONE)
#define OPUS_BP_INCLUDE_PANEL_16_DONE
namespace opus_bpreshuffle_gfx950::panel::schedule_16 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_11;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0, bool AScaleDma = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_supply_traits_gfx950 : BaseTraits {
using BASE_TRAITS = BaseTraits;
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static_assert((StaticN == 0) == (StaticStrideK == 0));
static_assert(StaticN == 0 || (StaticN > 0 && StaticN % 128 == 0 && StaticN % BaseTraits::B_N == 0));
static_assert(StaticStrideK == 0 || (StaticStrideK > 0 && StaticStrideK <= BaseTraits::MAX_K &&
StaticStrideK % BaseTraits::B_K == 0));
static_assert(StaticStrideK == 0 || BaseTraits::FIXED_K == 0 || StaticStrideK == BaseTraits::FIXED_K);
static_assert(!AScaleDma || !BaseTraits::PACKED_SCALES,
"Raw SFA DMA requires the LDS four-wave kernel");
};
#else

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0, bool AScaleDma = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_4wave_supply_traits_gfx950 : BaseTraits {
using BASE_TRAITS = BaseTraits;
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static_assert((StaticN == 0) == (StaticStrideK == 0));
static_assert(StaticN == 0 || (StaticN > 0 && StaticN % 128 == 0 && StaticN % BaseTraits::B_N == 0));
static_assert(StaticStrideK == 0 || (StaticStrideK > 0 && StaticStrideK <= BaseTraits::MAX_K &&
StaticStrideK % BaseTraits::B_K == 0));
static_assert(StaticStrideK == 0 || BaseTraits::FIXED_K == 0 || StaticStrideK == BaseTraits::FIXED_K);
static_assert(!AScaleDma || !BaseTraits::PACKED_SCALES,
"Raw SFA DMA requires the LDS four-wave kernel");
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PANEL_64) && !defined(OPUS_BP_INCLUDE_PANEL_64_DONE)
#define OPUS_BP_INCLUDE_PANEL_64_DONE
namespace opus_bpreshuffle_gfx950::panel::schedule_64 {
using namespace opus_bpreshuffle_gfx950::panel::schedule_12;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0,
bool AScaleDma = false, bool SfaWaveorder = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_panel8_supply_traits_gfx950 : BaseTraits {
using BASE_TRAITS = BaseTraits;
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static constexpr bool SFA_WAVEORDER = SfaWaveorder;
static_assert(StaticN == 0 && StaticStrideK == 0,
"Panel supply supports SFA DMA or wave ordering");
static_assert(!AScaleDma || !SfaWaveorder,
"SFA DMA and wave ordering are separate supply configurations");
static_assert(!SfaWaveorder || BaseTraits::INTERLEAVED);
};
#else

// SPDX-License-Identifier: Apache-2.0

template<class BaseTraits, int StaticN = 0, int StaticStrideK = 0,
bool AScaleDma = false, bool SfaWaveorder = false>
struct opus_gemm_blockscale_mxscale_bpreshuffle_panel8_supply_traits_gfx950 : BaseTraits {
using BASE_TRAITS = BaseTraits;
static constexpr int STATIC_N = StaticN;
static constexpr int STATIC_STRIDE_K = StaticStrideK;
static constexpr bool A_SCALE_DMA = AScaleDma;
static constexpr bool SFA_WAVEORDER = SfaWaveorder;
static_assert(StaticN == 0 && StaticStrideK == 0,
"Panel supply supports SFA DMA or wave ordering");
static_assert(!AScaleDma || !SfaWaveorder,
"SFA DMA and wave ordering are separate supply configurations");
static_assert(!SfaWaveorder || BaseTraits::INTERLEAVED);
};

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_PANEL_207) && !defined(OPUS_BP_INCLUDE_PANEL_207_DONE)
#define OPUS_BP_INCLUDE_PANEL_207_DONE
namespace opus_bpreshuffle_gfx950::panel::schedule_207 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0

template<int TileM, int ScalePanel = 32, int FixedK = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_2wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=128;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 2;
static constexpr int B_M = TileM;
static constexpr int FIXED_K = FixedK;
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
static constexpr int SCALE_PANEL = ScalePanel;
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
static_assert(SCALE_PANEL > 0);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % B_K == 0));
static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * MIN_WGS_PER_CU <= 160 * 1024);
};
#else

// SPDX-License-Identifier: Apache-2.0

template<int TileM, int ScalePanel = 32, int FixedK = 0>
struct opus_gemm_blockscale_mxscale_bpreshuffle_2wave_traits_gfx950 {
static constexpr int BLOCK_SIZE=128;
static constexpr int WARP_SIZE = 64;
static constexpr int NUM_WAVES = BLOCK_SIZE / WARP_SIZE;
static constexpr int MIN_WGS_PER_CU = 2;
static constexpr int B_M = TileM;
static constexpr int FIXED_K = FixedK;
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
static constexpr int SCALE_PANEL = ScalePanel;
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
static_assert(SCALE_PANEL > 0);
static_assert(FixedK == 0 || (FixedK > 0 && FixedK <= 16384 && FixedK % B_K == 0));
static_assert(((LDS_BYTES + 1279) / 1280 * 1280) * MIN_WGS_PER_CU <= 160 * 1024);
};

#endif
}
#endif
