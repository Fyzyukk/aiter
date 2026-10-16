#ifndef OPUS_BP_QUADRANT_PIPELINE_BASE
#define OPUS_BP_QUADRANT_PIPELINE_BASE
// SPDX-License-Identifier: Apache-2.0

#include "opus_gemm_blockscale_mxscale_bpreshuffle_support_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_quadrant_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_traits_tiled_gfx950.cuh"
#include "opus_gemm_blockscale_mxscale_bpreshuffle_pipeline_tiled_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)

namespace opus_bpreshuffle_gfx950::quadrant::schedule_1 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "kid 9000 requires yuyzhang512/llvm-project branch amdgpu-pin-op-dst"
#else
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
return result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
constexpr int threads_k = T::B_K / T::VEC_A;
constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
constexpr auto ga_block_shape = opus::make_tuple(
opus::number<T::HALF_B_M / threads_m_per_block>{},
opus::number<T::T_N>{},
opus::number<threads_m_per_wave>{},
opus::number<T::T_M>{},
opus::number<threads_k>{},
opus::number<T::VEC_A>{});
constexpr auto ga_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_A>(
ga_block_shape,
opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,lane_id % threads_k}));
}
template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
constexpr auto sa_block_shape = opus::make_tuple(
opus::number<T::smem_m_rep / num_waves>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::VEC_A>{});
constexpr auto sa_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sa_block_shape,
opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
constexpr int lanes_n = T::W_N;
constexpr int lanes_k = T::WARP_SIZE / lanes_n;
constexpr int k_per_wave_load = lanes_k * T::VEC_B;
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto gb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto gb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
gb_block_shape,
opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto sb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::VEC_B>{});
constexpr auto sb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sb_block_shape,
opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M / T::T_M>{},
opus::number<T::E_K>{},
opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
constexpr auto rb_block_shape = opus::make_tuple(
opus::number<T::E_N>{},
opus::number<T::T_N>{},
opus::number<T::E_K>{},
opus::number<k_vectors>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto rb_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
rb_block_shape,
opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SCALE_A>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SF>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_M>{},
opus::number<T::W_M>{},
opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<1>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
template<class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((kargs.m | kargs.n) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
auto g_a = [&] {
if constexpr (T::PAD_M) {
const unsigned int a_bytes = static_cast<unsigned int>((kargs.m - row) * kargs.stride_a);
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, a_bytes);
} else {
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a);
}
}();
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
auto g_c = [&] {
if constexpr (T::PAD_M) {
const unsigned int c_bytes = static_cast<unsigned int>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C));
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col, c_bytes);
} else {
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
}
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row, static_cast<unsigned int>(kargs.stride_sfa_batch - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * kargs.stride_sfb, static_cast<unsigned int>(kargs.stride_sfb));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * kargs.stride_a + (lane_id % a_k_lanes) * T::VEC_A : wave_id_m * T::HALF_B_N * kargs.stride_b + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * kargs.stride_a;
const int matrix_odd_stride = prefetches_a ? kargs.stride_a : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (T::PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = static_cast<unsigned int>(kargs.k) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (T::PAD_M) {
sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < kargs.m) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
} else {
if (first_k_group < scale_k_groups) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = static_cast<unsigned int>(kargs.k) / T::B_K;
if constexpr (T::PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma unroll 2
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 3>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
mma_scale_group<T, 6, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
int epilogue_lane_id = lane_id;
asm volatile("" : "+v"(epilogue_lane_id));
auto p_coord_c = opus::make_tuple(wave_id_m, epilogue_lane_id % mma.grpn_c, wave_id_n, epilogue_lane_id / mma.grpn_c);
auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(c_lds_row_stride_elems, 1_I), p_coord_c);
auto c_offset = [&](int half_tile_m, int half_tile_n) { return half_tile_m * T::HALF_B_M * c_lds_row_stride_elems + half_tile_n * T::HALF_B_N; };
const int output_thread_id = wave_id * T::WARP_SIZE + epilogue_lane_id;
auto copy_output_bf16 = [&](int half_m, int half_n, int copy_index) {
const int linear = output_thread_id * 8 + copy_index * T::BLOCK_SIZE * 8;
const int output_row = linear / T::HALF_B_N + half_m * T::HALF_B_M;
const int output_col = linear % T::HALF_B_N + half_n * T::HALF_B_N;
const auto value = load<8>(s_c, output_row * c_lds_row_stride_elems + output_col);
if constexpr (T::PAD_M) {
if (row + output_row < kargs.m) {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
} else {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
};
const auto gc_offsets = opus::layout_to_offsets<T::VEC_C>(u_gc);
auto stage_output_row = [&](auto index_i, int half_m, int half_n,
const AccFragment& c0, const AccFragment& c1,
const AccFragment& c2, const AccFragment& c3) {
constexpr int index = decltype(index_i)::value;
constexpr int n_repeat_stride = T::T_N * T::W_N;
const int row_offset = gc_offsets[index] + c_offset(half_m, half_n);
store<T::VEC_C>(s_c, cast<D_C>(c0), row_offset);
store<T::VEC_C>(s_c, cast<D_C>(c1), row_offset + n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c2), row_offset + 2 * n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c3), row_offset + 3 * n_repeat_stride);
};
auto store_output_bf16 = [&](auto half_m_i, auto half_n_i) {
constexpr int half_m = decltype(half_m_i)::value;
constexpr int half_n = decltype(half_n_i)::value;
__builtin_amdgcn_sched_barrier(0);
if constexpr (!(half_m == 1 && half_n == 0)) {
s_waitcnt_vmcnt(0_I);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
opus::static_for<8>([&](auto copy_i) { copy_output_bf16(half_m, half_n, decltype(copy_i)::value); });
};
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
stage_output_row(opus::number<0>{}, 0, 0, c00[0], c00[1], c00[2], c00[3]);
stage_output_row(opus::number<4>{}, 0, 0, c00[4], c00[5], c00[6], c00[7]);
stage_output_row(opus::number<8>{}, 0, 0, c00[8], c00[9], c00[10], c00[11]);
stage_output_row(opus::number<12>{}, 0, 0, c00[12], c00[13], c00[14], c00[15]);
store_output_bf16(opus::number<0>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 1, 0, c10[0], c10[1], c10[2], c10[3]);
stage_output_row(opus::number<4>{}, 1, 0, c10[4], c10[5], c10[6], c10[7]);
stage_output_row(opus::number<8>{}, 1, 0, c10[8], c10[9], c10[10], c10[11]);
stage_output_row(opus::number<12>{}, 1, 0, c10[12], c10[13], c10[14], c10[15]);
store_output_bf16(opus::number<1>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 0, 1, c01[0], c01[1], c01[2], c01[3]);
stage_output_row(opus::number<4>{}, 0, 1, c01[4], c01[5], c01[6], c01[7]);
stage_output_row(opus::number<8>{}, 0, 1, c01[8], c01[9], c01[10], c01[11]);
stage_output_row(opus::number<12>{}, 0, 1, c01[12], c01[13], c01[14], c01[15]);
store_output_bf16(opus::number<0>{}, opus::number<1>{});
stage_output_row(opus::number<0>{}, 1, 1, c11[0], c11[1], c11[2], c11[3]);
stage_output_row(opus::number<4>{}, 1, 1, c11[4], c11[5], c11[6], c11[7]);
stage_output_row(opus::number<8>{}, 1, 1, c11[8], c11[9], c11[10], c11[11]);
stage_output_row(opus::number<12>{}, 1, 1, c11[12], c11[13], c11[14], c11[15]);
store_output_bf16(opus::number<1>{}, opus::number<1>{});
}

#endif
#endif
}

namespace opus_bpreshuffle_gfx950::quadrant::schedule_2 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "kid 9000 requires yuyzhang512/llvm-project branch amdgpu-pin-op-dst"
#else
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
return result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
constexpr int threads_k = T::B_K / T::VEC_A;
constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
constexpr auto ga_block_shape = opus::make_tuple(
opus::number<T::HALF_B_M / threads_m_per_block>{},
opus::number<T::T_N>{},
opus::number<threads_m_per_wave>{},
opus::number<T::T_M>{},
opus::number<threads_k>{},
opus::number<T::VEC_A>{});
constexpr auto ga_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_A>(
ga_block_shape,
opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,lane_id % threads_k}));
}
template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
constexpr auto sa_block_shape = opus::make_tuple(
opus::number<T::smem_m_rep / num_waves>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::VEC_A>{});
constexpr auto sa_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sa_block_shape,
opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
constexpr int lanes_n = T::W_N;
constexpr int lanes_k = T::WARP_SIZE / lanes_n;
constexpr int k_per_wave_load = lanes_k * T::VEC_B;
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto gb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto gb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
gb_block_shape,
opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto sb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::VEC_B>{});
constexpr auto sb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sb_block_shape,
opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M / T::T_M>{},
opus::number<T::E_K>{},
opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
constexpr auto rb_block_shape = opus::make_tuple(
opus::number<T::E_N>{},
opus::number<T::T_N>{},
opus::number<T::E_K>{},
opus::number<k_vectors>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto rb_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
rb_block_shape,
opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SCALE_A>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SF>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_M>{},
opus::number<T::W_M>{},
opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<1>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
template<class Traits>
__device__ __forceinline__  void actual_pin_097_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((40960 | 7168) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
auto g_a = [&] {
if constexpr (T::PAD_M) {
const unsigned int a_bytes = static_cast<unsigned int>((40960 - row) * 768);
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * 31457280 + row * 768, a_bytes);
} else {
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * 31457280 + row * 768);
}
}();
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * 5505024 + col * 768);
auto g_c = [&] {
if constexpr (T::PAD_M) {
const unsigned int c_bytes = static_cast<unsigned int>(((40960 - row) * 7168 - col) * sizeof(D_C));
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * 293601280 + row * 7168 + col, c_bytes);
} else {
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * 293601280 + row * 7168 + col);
}
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * 245760 + row, static_cast<unsigned int>(245760 - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * 336 + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * 6, static_cast<unsigned int>(6));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, 768);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, 768);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, 40960);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * 768 + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * 768 + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * 40960; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * 768 + (lane_id % a_k_lanes) * T::VEC_A : wave_id_m * T::HALF_B_N * 768 + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * 768;
const int matrix_odd_stride = prefetches_a ? 768 : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (T::PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = static_cast<unsigned int>(768) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (T::PAD_M) {
sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < 40960) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
} else {
if (first_k_group < scale_k_groups) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = static_cast<unsigned int>(768) / T::B_K;
if constexpr (T::PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma unroll 2
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 3>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
mma_scale_group<T, 6, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
auto write_half = [&](const auto& accumulators,int half_m,int half_n) {
const int side=(lane_id/16)%2;
static_for<T::E_M*T::E_N/2>([&](auto pi) {
constexpr int ci=decltype(pi)::value*2;
const auto first=__builtin_bit_cast(vector_t<unsigned,2>,cast<D_C>(accumulators[ci]));
const auto second=__builtin_bit_cast(vector_t<unsigned,2>,cast<D_C>(accumulators[ci+1]));
const auto p0=__builtin_amdgcn_permlane16_swap(first[0],second[0],false,true);
const auto p1=__builtin_amdgcn_permlane16_swap(first[1],second[1],false,true);
const vector_t<unsigned,4> packed{p0[0],p1[0],p0[1],p1[1]};
const int frag=ci+side;
const int rr=half_m*T::HALF_B_M+((frag/T::E_N)*T::T_M+wave_id_m)*16+lane_id%16;
const int nn=half_n*T::HALF_B_N+((frag%T::E_N)*T::T_N+wave_id_n)*16+(lane_id/32)*8;
if(row+rr<40960)store<8>(g_c,__builtin_bit_cast(vector_t<D_C,8>,packed),rr*7168+nn,0,number<2>{});
});
};
write_half(c00,0,0);
write_half(c10,1,0);
write_half(c01,0,1);
write_half(c11,1,1);
s_waitcnt_vmcnt(0_I);
}
#endif
#endif
}

namespace opus_bpreshuffle_gfx950::quadrant::schedule_3 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_0;
using namespace opus_bpreshuffle_gfx950::tiled::schedule_1;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "The pinned bpreshuffle pipeline requires amdgpu-pin-op-dst"
#else
namespace opus_gemm_mxscale_bpreshuffle_pin_detail {
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
}
template<class Traits>
__device__ __forceinline__  void actual_pin_155_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using namespace opus_gemm_mxscale_bpreshuffle_pin_detail;
static_assert(T::SCHEDULE >= 0 && T::SCHEDULE <= 2);
constexpr bool PAD_M = T::SCHEDULE != 0;
constexpr int loop_unroll = T::SCHEDULE == 2 ? 1 : T::LOOP_UNROLL;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((2048 | 6144) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
auto g_a = [&] {
auto ptr = reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * 14680064 + row * 7168;
if constexpr (PAD_M)
return make_gmem(ptr, static_cast<unsigned int>((2048 - row) * 7168));
else
return make_gmem(ptr);
}();
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * 44040192 + col * 7168);
auto g_c = [&] {
auto ptr = reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * 12582912 + row * 6144 + col;
if constexpr (PAD_M)
return make_gmem(ptr, static_cast<unsigned int>(((2048 - row) * 6144 - col) * sizeof(D_C)));
else
return make_gmem(ptr);
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * 114688 + row, static_cast<unsigned int>(114688 - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * 2688 + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * 56, static_cast<unsigned int>(56));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, 7168);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, 7168);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, 2048);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * 7168 + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * 7168 + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * 2048; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * 7168 + (lane_id % a_k_lanes) * T::VEC_A : wave_id_m * T::HALF_B_N * 7168 + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * 7168;
const int matrix_odd_stride = prefetches_a ? 7168 : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = T::FIXED_K ? T::FIXED_K / T::GROUP_K : static_cast<unsigned int>(7168) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (PAD_M || T::RESET_SFA_BEFORE_LOAD) sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
if constexpr (PAD_M) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < 2048)
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
} else {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES && lane_id < T::SCALE_PANEL_K_CAPACITY) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES && lane_id < T::SCALE_PANEL_K_CAPACITY) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = T::FIXED_K ? T::FIXED_K / T::B_K : static_cast<unsigned int>(7168) / T::B_K;
if constexpr (PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES && lane_id < T::SCALE_PANEL_K_CAPACITY) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES && lane_id < T::SCALE_PANEL_K_CAPACITY) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma clang loop unroll_count(loop_unroll)
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 3>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
mma_scale_group<T, 6, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
auto write_half = [&](const auto& accumulators,int half_m,int half_n) {
const int side=(lane_id/16)%2;
static_for<T::E_M*T::E_N/2>([&](auto pi) {
constexpr int ci=decltype(pi)::value*2;
const auto first=__builtin_bit_cast(vector_t<unsigned,2>,cast<D_C>(accumulators[ci]));
const auto second=__builtin_bit_cast(vector_t<unsigned,2>,cast<D_C>(accumulators[ci+1]));
const auto p0=__builtin_amdgcn_permlane16_swap(first[0],second[0],false,true);
const auto p1=__builtin_amdgcn_permlane16_swap(first[1],second[1],false,true);
const vector_t<unsigned,4> packed{p0[0],p1[0],p0[1],p1[1]};
const int frag=ci+side;
const int rr=half_m*T::HALF_B_M+((frag/T::E_N)*T::T_M+wave_id_m)*16+lane_id%16;
const int nn=half_n*T::HALF_B_N+((frag%T::E_N)*T::T_N+wave_id_n)*16+(lane_id/32)*8;
if(row+rr<2048)store<8>(g_c,__builtin_bit_cast(vector_t<D_C,8>,packed),rr*6144+nn,0,number<2>{});
});
};
write_half(c00,0,0);
write_half(c10,1,0);
write_half(c01,0,1);
write_half(c11,1,1);
s_waitcnt_vmcnt(0_I);
}
#endif
#endif
}

namespace opus_bpreshuffle_gfx950::quadrant::schedule_4 {
using namespace opus_bpreshuffle_gfx950::tiled::schedule_0;
using namespace opus_bpreshuffle_gfx950::tiled::schedule_1;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "The pinned bpreshuffle pipeline requires amdgpu-pin-op-dst"
#else
namespace opus_gemm_mxscale_bpreshuffle_pin_detail {
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
}
template<class Traits>
__device__ __forceinline__  void actual_pin_176_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using namespace opus_gemm_mxscale_bpreshuffle_pin_detail;
static_assert(T::SCHEDULE >= 0 && T::SCHEDULE <= 2);
constexpr bool PAD_M = T::SCHEDULE != 0;
constexpr int loop_unroll = T::SCHEDULE == 2 ? 1 : T::LOOP_UNROLL;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((1792 | 7168) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
auto g_a = [&] {
auto ptr = reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * 29360128 + row * 16384;
if constexpr (PAD_M)
return make_gmem(ptr, static_cast<unsigned int>((1792 - row) * 16384));
else
return make_gmem(ptr);
}();
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * 117440512 + col * 16384);
auto g_c = [&] {
auto ptr = reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * 12845056 + row * 7168 + col;
if constexpr (PAD_M)
return make_gmem(ptr, static_cast<unsigned int>(((1792 - row) * 7168 - col) * sizeof(D_C)));
else
return make_gmem(ptr);
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * 229376 + row, static_cast<unsigned int>(229376 - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * 7168 + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * 128, static_cast<unsigned int>(128));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, 16384);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, 16384);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, 1792);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * 16384 + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * 16384 + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * 1792; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * 16384 + (lane_id % a_k_lanes) * T::VEC_A : wave_id_m * T::HALF_B_N * 16384 + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * 16384;
const int matrix_odd_stride = prefetches_a ? 16384 : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = T::FIXED_K ? T::FIXED_K / T::GROUP_K : static_cast<unsigned int>(16384) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (PAD_M || T::RESET_SFA_BEFORE_LOAD) sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
if constexpr (PAD_M) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < 1792)
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
} else {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES && lane_id < T::SCALE_PANEL_K_CAPACITY) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES && lane_id < T::SCALE_PANEL_K_CAPACITY) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = T::FIXED_K ? T::FIXED_K / T::B_K : static_cast<unsigned int>(16384) / T::B_K;
if constexpr (PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES && lane_id < T::SCALE_PANEL_K_CAPACITY) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES && lane_id < T::SCALE_PANEL_K_CAPACITY) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma clang loop unroll_count(loop_unroll)
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 5>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
auto write_half = [&](const auto& accumulators,int half_m,int half_n) {
const int side=(lane_id/16)%2;
static_for<T::E_M*T::E_N/2>([&](auto pi) {
constexpr int ci=decltype(pi)::value*2;
const auto first=__builtin_bit_cast(vector_t<unsigned,2>,cast<D_C>(accumulators[ci]));
const auto second=__builtin_bit_cast(vector_t<unsigned,2>,cast<D_C>(accumulators[ci+1]));
const auto p0=__builtin_amdgcn_permlane16_swap(first[0],second[0],false,true);
const auto p1=__builtin_amdgcn_permlane16_swap(first[1],second[1],false,true);
const vector_t<unsigned,4> packed{p0[0],p1[0],p0[1],p1[1]};
const int frag=ci+side;
const int rr=half_m*T::HALF_B_M+((frag/T::E_N)*T::T_M+wave_id_m)*16+lane_id%16;
const int nn=half_n*T::HALF_B_N+((frag%T::E_N)*T::T_N+wave_id_n)*16+(lane_id/32)*8;
if(row+rr<1792)store<8>(g_c,__builtin_bit_cast(vector_t<D_C,8>,packed),rr*7168+nn,0,number<2>{});
});
};
write_half(c00,0,0);
write_half(c10,1,0);
write_half(c01,0,1);
write_half(c11,1,1);
s_waitcnt_vmcnt(0_I);
}
#endif
#endif
}

namespace opus_bpreshuffle_gfx950::quadrant::schedule_5 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "kid 9000 requires yuyzhang512/llvm-project branch amdgpu-pin-op-dst"
#else
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
return result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
constexpr int threads_k = T::B_K / T::VEC_A;
constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
constexpr auto ga_block_shape = opus::make_tuple(
opus::number<T::HALF_B_M / threads_m_per_block>{},
opus::number<T::T_N>{},
opus::number<threads_m_per_wave>{},
opus::number<T::T_M>{},
opus::number<threads_k>{},
opus::number<T::VEC_A>{});
constexpr auto ga_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_A>(
ga_block_shape,
opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,lane_id % threads_k}));
}
template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
constexpr auto sa_block_shape = opus::make_tuple(
opus::number<T::smem_m_rep / num_waves>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::VEC_A>{});
constexpr auto sa_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sa_block_shape,
opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
constexpr int lanes_n = T::W_N;
constexpr int lanes_k = T::WARP_SIZE / lanes_n;
constexpr int k_per_wave_load = lanes_k * T::VEC_B;
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto gb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto gb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
gb_block_shape,
opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto sb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::VEC_B>{});
constexpr auto sb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sb_block_shape,
opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M / T::T_M>{},
opus::number<T::E_K>{},
opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
constexpr auto rb_block_shape = opus::make_tuple(
opus::number<T::E_N>{},
opus::number<T::T_N>{},
opus::number<T::E_K>{},
opus::number<k_vectors>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto rb_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
rb_block_shape,
opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SCALE_A>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SF>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_M>{},
opus::number<T::W_M>{},
opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<1>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
template<class Traits>
__device__ __forceinline__  void actual_pin_181_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((1984 | 7168) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
auto g_a = [&] {
if constexpr (T::PAD_M) {
const unsigned int a_bytes = static_cast<unsigned int>((1984 - row) * 16384);
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * 32505856 + row * 16384, a_bytes);
} else {
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * 32505856 + row * 16384);
}
}();
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * 117440512 + col * 16384);
auto g_c = [&] {
if constexpr (T::PAD_M) {
const unsigned int c_bytes = static_cast<unsigned int>(((1984 - row) * 7168 - col) * sizeof(D_C));
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * 14221312 + row * 7168 + col, c_bytes);
} else {
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * 14221312 + row * 7168 + col);
}
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * 253952 + row, static_cast<unsigned int>(253952 - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * 7168 + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * 128, static_cast<unsigned int>(128));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, 16384);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, 16384);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, 1984);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * 16384 + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * 16384 + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * 1984; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * 16384 + (lane_id % a_k_lanes) * T::VEC_A : wave_id_m * T::HALF_B_N * 16384 + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * 16384;
const int matrix_odd_stride = prefetches_a ? 16384 : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (T::PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = static_cast<unsigned int>(16384) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (T::PAD_M) {
sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < 1984) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
} else {
if (first_k_group < scale_k_groups) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = static_cast<unsigned int>(16384) / T::B_K;
if constexpr (T::PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma unroll 2
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 3>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
mma_scale_group<T, 6, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
auto write_half = [&](const auto& accumulators,int half_m,int half_n) {
const int side=(lane_id/16)%2;
static_for<T::E_M*T::E_N/2>([&](auto pi) {
constexpr int ci=decltype(pi)::value*2;
const auto first=__builtin_bit_cast(vector_t<unsigned,2>,cast<D_C>(accumulators[ci]));
const auto second=__builtin_bit_cast(vector_t<unsigned,2>,cast<D_C>(accumulators[ci+1]));
const auto p0=__builtin_amdgcn_permlane16_swap(first[0],second[0],false,true);
const auto p1=__builtin_amdgcn_permlane16_swap(first[1],second[1],false,true);
const vector_t<unsigned,4> packed{p0[0],p1[0],p0[1],p1[1]};
const int frag=ci+side;
const int rr=half_m*T::HALF_B_M+((frag/T::E_N)*T::T_M+wave_id_m)*16+lane_id%16;
const int nn=half_n*T::HALF_B_N+((frag%T::E_N)*T::T_N+wave_id_n)*16+(lane_id/32)*8;
if(row+rr<1984)store<8>(g_c,__builtin_bit_cast(vector_t<D_C,8>,packed),rr*7168+nn,0,number<2>{});
});
};
write_half(c00,0,0);
write_half(c10,1,0);
write_half(c01,0,1);
write_half(c11,1,1);
s_waitcnt_vmcnt(0_I);
}
#endif
#endif
}
#endif

#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_7) && !defined(OPUS_BP_INCLUDE_QUADRANT_7_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_7_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_7 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0



template<class Traits>
__global__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0



#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "kid 9000 requires yuyzhang512/llvm-project branch amdgpu-pin-op-dst"
#else
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
return result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
constexpr int threads_k = T::B_K / T::VEC_A;
constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
constexpr auto ga_block_shape = opus::make_tuple(
opus::number<T::HALF_B_M / threads_m_per_block>{},
opus::number<T::T_N>{},
opus::number<threads_m_per_wave>{},
opus::number<T::T_M>{},
opus::number<threads_k>{},
opus::number<T::VEC_A>{});
constexpr auto ga_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_A>(
ga_block_shape,
opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,lane_id % threads_k}));
}
template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
constexpr auto sa_block_shape = opus::make_tuple(
opus::number<T::smem_m_rep / num_waves>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::VEC_A>{});
constexpr auto sa_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sa_block_shape,
opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
constexpr int lanes_n = T::W_N;
constexpr int lanes_k = T::WARP_SIZE / lanes_n;
constexpr int k_per_wave_load = lanes_k * T::VEC_B;
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto gb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto gb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
gb_block_shape,
opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto sb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::VEC_B>{});
constexpr auto sb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sb_block_shape,
opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M / T::T_M>{},
opus::number<T::E_K>{},
opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
constexpr auto rb_block_shape = opus::make_tuple(
opus::number<T::E_N>{},
opus::number<T::T_N>{},
opus::number<T::E_K>{},
opus::number<k_vectors>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto rb_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
rb_block_shape,
opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SCALE_A>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SF>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_M>{},
opus::number<T::W_M>{},
opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<1>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
template<class Traits>
__global__ __launch_bounds__(256, 1) void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((kargs.m | kargs.n) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
auto g_a = [&] {
if constexpr (T::PAD_M) {
const unsigned int a_bytes = static_cast<unsigned int>((kargs.m - row) * kargs.stride_a);
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, a_bytes);
} else {
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a);
}
}();
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
auto g_c = [&] {
if constexpr (T::PAD_M) {
const unsigned int c_bytes = static_cast<unsigned int>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C));
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col, c_bytes);
} else {
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
}
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row, static_cast<unsigned int>(kargs.stride_sfa_batch - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * kargs.stride_sfb, static_cast<unsigned int>(kargs.stride_sfb));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * kargs.stride_a + (lane_id % a_k_lanes) * T::VEC_A : wave_id_m * T::HALF_B_N * kargs.stride_b + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * kargs.stride_a;
const int matrix_odd_stride = prefetches_a ? kargs.stride_a : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (T::PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = static_cast<unsigned int>(kargs.k) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (T::PAD_M) {
sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < kargs.m) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
} else {
if (first_k_group < scale_k_groups) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = static_cast<unsigned int>(kargs.k) / T::B_K;
if constexpr (T::PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma unroll 2
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 3>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
mma_scale_group<T, 6, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
int epilogue_lane_id = lane_id;
asm volatile("" : "+v"(epilogue_lane_id));
auto p_coord_c = opus::make_tuple(wave_id_m, epilogue_lane_id % mma.grpn_c, wave_id_n, epilogue_lane_id / mma.grpn_c);
auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(c_lds_row_stride_elems, 1_I), p_coord_c);
auto c_offset = [&](int half_tile_m, int half_tile_n) { return half_tile_m * T::HALF_B_M * c_lds_row_stride_elems + half_tile_n * T::HALF_B_N; };
const int output_thread_id = wave_id * T::WARP_SIZE + epilogue_lane_id;
auto copy_output_bf16 = [&](int half_m, int half_n, int copy_index) {
const int linear = output_thread_id * 8 + copy_index * T::BLOCK_SIZE * 8;
const int output_row = linear / T::HALF_B_N + half_m * T::HALF_B_M;
const int output_col = linear % T::HALF_B_N + half_n * T::HALF_B_N;
const auto value = load<8>(s_c, output_row * c_lds_row_stride_elems + output_col);
if constexpr (T::PAD_M) {
if (row + output_row < kargs.m) {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
} else {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
};
const auto gc_offsets = opus::layout_to_offsets<T::VEC_C>(u_gc);
auto stage_output_row = [&](auto index_i, int half_m, int half_n,
const AccFragment& c0, const AccFragment& c1,
const AccFragment& c2, const AccFragment& c3) {
constexpr int index = decltype(index_i)::value;
constexpr int n_repeat_stride = T::T_N * T::W_N;
const int row_offset = gc_offsets[index] + c_offset(half_m, half_n);
store<T::VEC_C>(s_c, cast<D_C>(c0), row_offset);
store<T::VEC_C>(s_c, cast<D_C>(c1), row_offset + n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c2), row_offset + 2 * n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c3), row_offset + 3 * n_repeat_stride);
};
auto store_output_bf16 = [&](auto half_m_i, auto half_n_i) {
constexpr int half_m = decltype(half_m_i)::value;
constexpr int half_n = decltype(half_n_i)::value;
__builtin_amdgcn_sched_barrier(0);
if constexpr (!(half_m == 1 && half_n == 0)) {
s_waitcnt_vmcnt(0_I);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
opus::static_for<8>([&](auto copy_i) { copy_output_bf16(half_m, half_n, decltype(copy_i)::value); });
};
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
stage_output_row(opus::number<0>{}, 0, 0, c00[0], c00[1], c00[2], c00[3]);
stage_output_row(opus::number<4>{}, 0, 0, c00[4], c00[5], c00[6], c00[7]);
stage_output_row(opus::number<8>{}, 0, 0, c00[8], c00[9], c00[10], c00[11]);
stage_output_row(opus::number<12>{}, 0, 0, c00[12], c00[13], c00[14], c00[15]);
store_output_bf16(opus::number<0>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 1, 0, c10[0], c10[1], c10[2], c10[3]);
stage_output_row(opus::number<4>{}, 1, 0, c10[4], c10[5], c10[6], c10[7]);
stage_output_row(opus::number<8>{}, 1, 0, c10[8], c10[9], c10[10], c10[11]);
stage_output_row(opus::number<12>{}, 1, 0, c10[12], c10[13], c10[14], c10[15]);
store_output_bf16(opus::number<1>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 0, 1, c01[0], c01[1], c01[2], c01[3]);
stage_output_row(opus::number<4>{}, 0, 1, c01[4], c01[5], c01[6], c01[7]);
stage_output_row(opus::number<8>{}, 0, 1, c01[8], c01[9], c01[10], c01[11]);
stage_output_row(opus::number<12>{}, 0, 1, c01[12], c01[13], c01[14], c01[15]);
store_output_bf16(opus::number<0>{}, opus::number<1>{});
stage_output_row(opus::number<0>{}, 1, 1, c11[0], c11[1], c11[2], c11[3]);
stage_output_row(opus::number<4>{}, 1, 1, c11[4], c11[5], c11[6], c11[7]);
stage_output_row(opus::number<8>{}, 1, 1, c11[8], c11[9], c11[10], c11[11]);
stage_output_row(opus::number<12>{}, 1, 1, c11[12], c11[13], c11[14], c11[15]);
store_output_bf16(opus::number<1>{}, opus::number<1>{});
}
#endif
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_8) && !defined(OPUS_BP_INCLUDE_QUADRANT_8_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_8_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_8 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0



template<class Traits>
__global__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0



#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "kid 9000 requires yuyzhang512/llvm-project branch amdgpu-pin-op-dst"
#else
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
return result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
constexpr int threads_k = T::B_K / T::VEC_A;
constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
constexpr auto ga_block_shape = opus::make_tuple(
opus::number<T::HALF_B_M / threads_m_per_block>{},
opus::number<T::T_N>{},
opus::number<threads_m_per_wave>{},
opus::number<T::T_M>{},
opus::number<threads_k>{},
opus::number<T::VEC_A>{});
constexpr auto ga_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_A>(
ga_block_shape,
opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,lane_id % threads_k}));
}
template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
constexpr auto sa_block_shape = opus::make_tuple(
opus::number<T::smem_m_rep / num_waves>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::VEC_A>{});
constexpr auto sa_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sa_block_shape,
opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
constexpr int lanes_n = T::W_N;
constexpr int lanes_k = T::WARP_SIZE / lanes_n;
constexpr int k_per_wave_load = lanes_k * T::VEC_B;
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto gb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto gb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
gb_block_shape,
opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto sb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::VEC_B>{});
constexpr auto sb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sb_block_shape,
opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M / T::T_M>{},
opus::number<T::E_K>{},
opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
constexpr auto rb_block_shape = opus::make_tuple(
opus::number<T::E_N>{},
opus::number<T::T_N>{},
opus::number<T::E_K>{},
opus::number<k_vectors>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto rb_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
rb_block_shape,
opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SCALE_A>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SF>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_M>{},
opus::number<T::W_M>{},
opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<1>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
template<class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((kargs.m | kargs.n) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch,
static_cast<unsigned int>(kargs.m * kargs.stride_a));
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch, static_cast<unsigned int>(kargs.n * kargs.stride_b));
auto g_c = [&] {
if constexpr (T::PAD_M) {
const unsigned int c_bytes = static_cast<unsigned int>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C));
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col, c_bytes);
} else {
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
}
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row, static_cast<unsigned int>(kargs.stride_sfa_batch - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * kargs.stride_sfb, static_cast<unsigned int>(kargs.stride_sfb));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return (row + half_tile_m * T::HALF_B_M) * kargs.stride_a + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return (col + half_tile_n * T::HALF_B_N) * kargs.stride_b + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (row + wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * kargs.stride_a + (lane_id % a_k_lanes) * T::VEC_A : (col + wave_id_m * T::HALF_B_N) * kargs.stride_b + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * kargs.stride_a;
const int matrix_odd_stride = prefetches_a ? kargs.stride_a : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (T::PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = static_cast<unsigned int>(kargs.k) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (T::PAD_M) {
sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < kargs.m) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
} else {
if (first_k_group < scale_k_groups) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = static_cast<unsigned int>(kargs.k) / T::B_K;
if constexpr (T::PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma unroll 2
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 5>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
int epilogue_lane_id = lane_id;
asm volatile("" : "+v"(epilogue_lane_id));
auto p_coord_c = opus::make_tuple(wave_id_m, epilogue_lane_id % mma.grpn_c, wave_id_n, epilogue_lane_id / mma.grpn_c);
auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(c_lds_row_stride_elems, 1_I), p_coord_c);
auto c_offset = [&](int half_tile_m, int half_tile_n) { return half_tile_m * T::HALF_B_M * c_lds_row_stride_elems + half_tile_n * T::HALF_B_N; };
const int output_thread_id = wave_id * T::WARP_SIZE + epilogue_lane_id;
auto copy_output_bf16 = [&](int half_m, int half_n, int copy_index) {
const int linear = output_thread_id * 8 + copy_index * T::BLOCK_SIZE * 8;
const int output_row = linear / T::HALF_B_N + half_m * T::HALF_B_M;
const int output_col = linear % T::HALF_B_N + half_n * T::HALF_B_N;
const auto value = load<8>(s_c, output_row * c_lds_row_stride_elems + output_col);
if constexpr (T::PAD_M) {
if (row + output_row < kargs.m) {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
} else {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
};
const auto gc_offsets = opus::layout_to_offsets<T::VEC_C>(u_gc);
auto stage_output_row = [&](auto index_i, int half_m, int half_n,
const AccFragment& c0, const AccFragment& c1,
const AccFragment& c2, const AccFragment& c3) {
constexpr int index = decltype(index_i)::value;
constexpr int n_repeat_stride = T::T_N * T::W_N;
const int row_offset = gc_offsets[index] + c_offset(half_m, half_n);
store<T::VEC_C>(s_c, cast<D_C>(c0), row_offset);
store<T::VEC_C>(s_c, cast<D_C>(c1), row_offset + n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c2), row_offset + 2 * n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c3), row_offset + 3 * n_repeat_stride);
};
auto store_output_bf16 = [&](auto half_m_i, auto half_n_i) {
constexpr int half_m = decltype(half_m_i)::value;
constexpr int half_n = decltype(half_n_i)::value;
opus::static_for<8>([&](auto copy_i) { copy_output_bf16(half_m, half_n, decltype(copy_i)::value); });
};
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
stage_output_row(opus::number<0>{}, 0, 0, c00[0], c00[1], c00[2], c00[3]);
stage_output_row(opus::number<4>{}, 0, 0, c00[4], c00[5], c00[6], c00[7]);
stage_output_row(opus::number<8>{}, 0, 0, c00[8], c00[9], c00[10], c00[11]);
stage_output_row(opus::number<12>{}, 0, 0, c00[12], c00[13], c00[14], c00[15]);
stage_output_row(opus::number<0>{}, 1, 0, c10[0], c10[1], c10[2], c10[3]);
stage_output_row(opus::number<4>{}, 1, 0, c10[4], c10[5], c10[6], c10[7]);
stage_output_row(opus::number<8>{}, 1, 0, c10[8], c10[9], c10[10], c10[11]);
stage_output_row(opus::number<12>{}, 1, 0, c10[12], c10[13], c10[14], c10[15]);
stage_output_row(opus::number<0>{}, 0, 1, c01[0], c01[1], c01[2], c01[3]);
stage_output_row(opus::number<4>{}, 0, 1, c01[4], c01[5], c01[6], c01[7]);
stage_output_row(opus::number<8>{}, 0, 1, c01[8], c01[9], c01[10], c01[11]);
stage_output_row(opus::number<12>{}, 0, 1, c01[12], c01[13], c01[14], c01[15]);
stage_output_row(opus::number<0>{}, 1, 1, c11[0], c11[1], c11[2], c11[3]);
stage_output_row(opus::number<4>{}, 1, 1, c11[4], c11[5], c11[6], c11[7]);
stage_output_row(opus::number<8>{}, 1, 1, c11[8], c11[9], c11[10], c11[11]);
stage_output_row(opus::number<12>{}, 1, 1, c11[12], c11[13], c11[14], c11[15]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
store_output_bf16(opus::number<0>{}, opus::number<0>{});
store_output_bf16(opus::number<1>{}, opus::number<0>{});
store_output_bf16(opus::number<0>{}, opus::number<1>{});
store_output_bf16(opus::number<1>{}, opus::number<1>{});
}
template<class Traits>
__global__ __launch_bounds__(256, 1) void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_body_gfx950<Traits>(kargs);
}

#endif
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_9) && !defined(OPUS_BP_INCLUDE_QUADRANT_9_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_9_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_9 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_8;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0








using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits = opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits_gfx950<true>;
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t byte_limit = 2147483647;
AITER_CHECK(m <= byte_limit / k && n <= byte_limit / k &&
m <= (byte_limit / sizeof(D_C)) / n,
entry, ": tensor byte extent exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ/Y must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = m * n;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 255) / 256;
const dim3 grid(n / 256, tiles_m);
opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits><<<
grid, dim3(256), 0, aiter::getCurrentHIPStream()>>>(args);
}
#else

// SPDX-License-Identifier: Apache-2.0
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)






#endif
#ifdef OPUS_FUSED_HOST_TU


#else

#endif
using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits = opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits_gfx950<true>;
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t byte_limit = 2147483647;
AITER_CHECK(m <= byte_limit / k && n <= byte_limit / k &&
m <= (byte_limit / sizeof(D_C)) / n,
entry, ": tensor byte extent exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ/Y must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = m * n;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 255) / 256;
const dim3 grid(n / 256, tiles_m);
opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits><<<
grid, dim3(256), 0, aiter::getCurrentHIPStream()>>>(args);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_10) && !defined(OPUS_BP_INCLUDE_QUADRANT_10_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_10_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_10 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_9;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 0, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M == 1856 && N == 7168 && K == 3072, "requires m_align");
AITER_CHECK(splitK == 1, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
#if !defined(__HIP_DEVICE_COMPILE__)
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 0, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M == 1856 && N == 7168 && K == 3072, "requires m_align");
AITER_CHECK(splitK == 1, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif

#endif
}
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template __global__ void opus_bpreshuffle_gfx950::quadrant::schedule_8::opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel<opus_bpreshuffle_gfx950::quadrant::schedule_6::opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits_gfx950<true>>(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950);
#endif

#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_11) && !defined(OPUS_BP_INCLUDE_QUADRANT_11_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_11_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_11 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0



template<class Traits>
__global__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0



#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "kid 9000 requires yuyzhang512/llvm-project branch amdgpu-pin-op-dst"
#else
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
return result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
constexpr int threads_k = T::B_K / T::VEC_A;
constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
constexpr auto ga_block_shape = opus::make_tuple(
opus::number<T::HALF_B_M / threads_m_per_block>{},
opus::number<T::T_N>{},
opus::number<threads_m_per_wave>{},
opus::number<T::T_M>{},
opus::number<threads_k>{},
opus::number<T::VEC_A>{});
constexpr auto ga_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_A>(
ga_block_shape,
opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,lane_id % threads_k}));
}
template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
constexpr auto sa_block_shape = opus::make_tuple(
opus::number<T::smem_m_rep / num_waves>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::VEC_A>{});
constexpr auto sa_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sa_block_shape,
opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
constexpr int lanes_n = T::W_N;
constexpr int lanes_k = T::WARP_SIZE / lanes_n;
constexpr int k_per_wave_load = lanes_k * T::VEC_B;
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto gb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto gb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
gb_block_shape,
opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto sb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::VEC_B>{});
constexpr auto sb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sb_block_shape,
opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M / T::T_M>{},
opus::number<T::E_K>{},
opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
constexpr auto rb_block_shape = opus::make_tuple(
opus::number<T::E_N>{},
opus::number<T::T_N>{},
opus::number<T::E_K>{},
opus::number<k_vectors>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto rb_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
rb_block_shape,
opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SCALE_A>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SF>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_M>{},
opus::number<T::W_M>{},
opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<1>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
template<class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((kargs.m | kargs.n) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch,
static_cast<unsigned int>(kargs.m * kargs.stride_a));
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch, static_cast<unsigned int>(kargs.n * kargs.stride_b));
auto g_c = [&] {
if constexpr (T::PAD_M) {
const unsigned int c_bytes = static_cast<unsigned int>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C));
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col, c_bytes);
} else {
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
}
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row, static_cast<unsigned int>(kargs.stride_sfa_batch - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * kargs.stride_sfb, static_cast<unsigned int>(kargs.stride_sfb));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return (row + half_tile_m * T::HALF_B_M) * kargs.stride_a + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return (col + half_tile_n * T::HALF_B_N) * kargs.stride_b + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (row + wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * kargs.stride_a + (lane_id % a_k_lanes) * T::VEC_A : (col + wave_id_m * T::HALF_B_N) * kargs.stride_b + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * kargs.stride_a;
const int matrix_odd_stride = prefetches_a ? kargs.stride_a : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (T::PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = static_cast<unsigned int>(kargs.k) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (T::PAD_M) {
sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < kargs.m) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
} else {
if (first_k_group < scale_k_groups) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = static_cast<unsigned int>(kargs.k) / T::B_K;
if constexpr (T::PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma unroll 2
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 5>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
int epilogue_lane_id = lane_id;
asm volatile("" : "+v"(epilogue_lane_id));
auto p_coord_c = opus::make_tuple(wave_id_m, epilogue_lane_id % mma.grpn_c, wave_id_n, epilogue_lane_id / mma.grpn_c);
auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(c_lds_row_stride_elems, 1_I), p_coord_c);
auto c_offset = [&](int half_tile_m, int half_tile_n) { return half_tile_m * T::HALF_B_M * c_lds_row_stride_elems + half_tile_n * T::HALF_B_N; };
const int output_thread_id = wave_id * T::WARP_SIZE + epilogue_lane_id;
auto copy_output_bf16 = [&](int half_m, int half_n, int copy_index) {
const int linear = output_thread_id * 8 + copy_index * T::BLOCK_SIZE * 8;
const int output_row = linear / T::HALF_B_N + half_m * T::HALF_B_M;
const int output_col = linear % T::HALF_B_N + half_n * T::HALF_B_N;
const auto value = load<8>(s_c, output_row * c_lds_row_stride_elems + output_col);
if constexpr (T::PAD_M) {
if (row + output_row < kargs.m) {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
} else {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
};
const auto gc_offsets = opus::layout_to_offsets<T::VEC_C>(u_gc);
auto stage_output_row = [&](auto index_i, int half_m, int half_n,
const AccFragment& c0, const AccFragment& c1,
const AccFragment& c2, const AccFragment& c3) {
constexpr int index = decltype(index_i)::value;
constexpr int n_repeat_stride = T::T_N * T::W_N;
const int row_offset = gc_offsets[index] + c_offset(half_m, half_n);
store<T::VEC_C>(s_c, cast<D_C>(c0), row_offset);
store<T::VEC_C>(s_c, cast<D_C>(c1), row_offset + n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c2), row_offset + 2 * n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c3), row_offset + 3 * n_repeat_stride);
};
auto store_output_bf16 = [&](auto half_m_i, auto half_n_i) {
constexpr int half_m = decltype(half_m_i)::value;
constexpr int half_n = decltype(half_n_i)::value;
__builtin_amdgcn_sched_barrier(0);
if constexpr (!(half_m == 1 && half_n == 0)) {
s_waitcnt_vmcnt(0_I);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
opus::static_for<8>([&](auto copy_i) { copy_output_bf16(half_m, half_n, decltype(copy_i)::value); });
};
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
stage_output_row(opus::number<0>{}, 0, 0, c00[0], c00[1], c00[2], c00[3]);
stage_output_row(opus::number<4>{}, 0, 0, c00[4], c00[5], c00[6], c00[7]);
stage_output_row(opus::number<8>{}, 0, 0, c00[8], c00[9], c00[10], c00[11]);
stage_output_row(opus::number<12>{}, 0, 0, c00[12], c00[13], c00[14], c00[15]);
store_output_bf16(opus::number<0>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 1, 0, c10[0], c10[1], c10[2], c10[3]);
stage_output_row(opus::number<4>{}, 1, 0, c10[4], c10[5], c10[6], c10[7]);
stage_output_row(opus::number<8>{}, 1, 0, c10[8], c10[9], c10[10], c10[11]);
stage_output_row(opus::number<12>{}, 1, 0, c10[12], c10[13], c10[14], c10[15]);
store_output_bf16(opus::number<1>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 0, 1, c01[0], c01[1], c01[2], c01[3]);
stage_output_row(opus::number<4>{}, 0, 1, c01[4], c01[5], c01[6], c01[7]);
stage_output_row(opus::number<8>{}, 0, 1, c01[8], c01[9], c01[10], c01[11]);
stage_output_row(opus::number<12>{}, 0, 1, c01[12], c01[13], c01[14], c01[15]);
store_output_bf16(opus::number<0>{}, opus::number<1>{});
stage_output_row(opus::number<0>{}, 1, 1, c11[0], c11[1], c11[2], c11[3]);
stage_output_row(opus::number<4>{}, 1, 1, c11[4], c11[5], c11[6], c11[7]);
stage_output_row(opus::number<8>{}, 1, 1, c11[8], c11[9], c11[10], c11[11]);
stage_output_row(opus::number<12>{}, 1, 1, c11[12], c11[13], c11[14], c11[15]);
store_output_bf16(opus::number<1>{}, opus::number<1>{});
}
template<class Traits>
__global__ __launch_bounds__(256, 1) void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_body_gfx950<Traits>(kargs);
}

#endif
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_12) && !defined(OPUS_BP_INCLUDE_QUADRANT_12_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_12_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_12 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_11;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0








using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits = opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits_gfx950<true>;
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t byte_limit = 2147483647;
AITER_CHECK(m <= byte_limit / k && n <= byte_limit / k &&
m <= (byte_limit / sizeof(D_C)) / n,
entry, ": tensor byte extent exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ/Y must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = m * n;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 255) / 256;
const dim3 grid(n / 256, tiles_m);
opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits><<<
grid, dim3(256), 0, aiter::getCurrentHIPStream()>>>(args);
}
#else

// SPDX-License-Identifier: Apache-2.0
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)






#endif
#ifdef OPUS_FUSED_HOST_TU


#else

#endif
using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits = opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits_gfx950<true>;
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t byte_limit = 2147483647;
AITER_CHECK(m <= byte_limit / k && n <= byte_limit / k &&
m <= (byte_limit / sizeof(D_C)) / n,
entry, ": tensor byte extent exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ/Y must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = m * n;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 255) / 256;
const dim3 grid(n / 256, tiles_m);
opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits><<<
grid, dim3(256), 0, aiter::getCurrentHIPStream()>>>(args);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_13) && !defined(OPUS_BP_INCLUDE_QUADRANT_13_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_13_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_13 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_12;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 0, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M == 1856 && N == 7168 && K == 3072, "requires m_align");
AITER_CHECK(splitK == 1, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
#if !defined(__HIP_DEVICE_COMPILE__)
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 0, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M == 1856 && N == 7168 && K == 3072, "requires m_align");
AITER_CHECK(splitK == 1, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif

#endif
}
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template __global__ void opus_bpreshuffle_gfx950::quadrant::schedule_11::opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel<opus_bpreshuffle_gfx950::quadrant::schedule_6::opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits_gfx950<true>>(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950);
#endif

#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_14) && !defined(OPUS_BP_INCLUDE_QUADRANT_14_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_14_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_14 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0



template<class Traits>
__global__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0



#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "kid 9000 requires yuyzhang512/llvm-project branch amdgpu-pin-op-dst"
#else
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
return result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
constexpr int threads_k = T::B_K / T::VEC_A;
constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
constexpr auto ga_block_shape = opus::make_tuple(
opus::number<T::HALF_B_M / threads_m_per_block>{},
opus::number<T::T_N>{},
opus::number<threads_m_per_wave>{},
opus::number<T::T_M>{},
opus::number<threads_k>{},
opus::number<T::VEC_A>{});
constexpr auto ga_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_A>(
ga_block_shape,
opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,lane_id % threads_k}));
}
template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
constexpr auto sa_block_shape = opus::make_tuple(
opus::number<T::smem_m_rep / num_waves>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::VEC_A>{});
constexpr auto sa_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sa_block_shape,
opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
constexpr int lanes_n = T::W_N;
constexpr int lanes_k = T::WARP_SIZE / lanes_n;
constexpr int k_per_wave_load = lanes_k * T::VEC_B;
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto gb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto gb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
gb_block_shape,
opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto sb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::VEC_B>{});
constexpr auto sb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sb_block_shape,
opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M / T::T_M>{},
opus::number<T::E_K>{},
opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
constexpr auto rb_block_shape = opus::make_tuple(
opus::number<T::E_N>{},
opus::number<T::T_N>{},
opus::number<T::E_K>{},
opus::number<k_vectors>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto rb_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
rb_block_shape,
opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SCALE_A>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SF>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_M>{},
opus::number<T::W_M>{},
opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<1>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
template<class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((kargs.m | kargs.n) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch,
static_cast<unsigned int>(kargs.m * kargs.stride_a));
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch, static_cast<unsigned int>(kargs.n * kargs.stride_b));
auto g_c = [&] {
if constexpr (T::PAD_M) {
const unsigned int c_bytes = static_cast<unsigned int>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C));
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col, c_bytes);
} else {
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
}
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row, static_cast<unsigned int>(kargs.stride_sfa_batch - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * kargs.stride_sfb, static_cast<unsigned int>(kargs.stride_sfb));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return (row + half_tile_m * T::HALF_B_M) * kargs.stride_a + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return (col + half_tile_n * T::HALF_B_N) * kargs.stride_b + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (row + wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * kargs.stride_a + (lane_id % a_k_lanes) * T::VEC_A : (col + wave_id_m * T::HALF_B_N) * kargs.stride_b + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * kargs.stride_a;
const int matrix_odd_stride = prefetches_a ? kargs.stride_a : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (T::PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = static_cast<unsigned int>(kargs.k) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (T::PAD_M) {
sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < kargs.m) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
} else {
if (first_k_group < scale_k_groups) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = static_cast<unsigned int>(kargs.k) / T::B_K;
if constexpr (T::PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma unroll 2
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 3>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
mma_scale_group<T, 6, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
int epilogue_lane_id = lane_id;
asm volatile("" : "+v"(epilogue_lane_id));
auto p_coord_c = opus::make_tuple(wave_id_m, epilogue_lane_id % mma.grpn_c, wave_id_n, epilogue_lane_id / mma.grpn_c);
auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(c_lds_row_stride_elems, 1_I), p_coord_c);
auto c_offset = [&](int half_tile_m, int half_tile_n) { return half_tile_m * T::HALF_B_M * c_lds_row_stride_elems + half_tile_n * T::HALF_B_N; };
const int output_thread_id = wave_id * T::WARP_SIZE + epilogue_lane_id;
auto copy_output_bf16 = [&](int half_m, int half_n, int copy_index) {
const int linear = output_thread_id * 8 + copy_index * T::BLOCK_SIZE * 8;
const int output_row = linear / T::HALF_B_N + half_m * T::HALF_B_M;
const int output_col = linear % T::HALF_B_N + half_n * T::HALF_B_N;
const auto value = load<8>(s_c, output_row * c_lds_row_stride_elems + output_col);
if constexpr (T::PAD_M) {
if (row + output_row < kargs.m) {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
} else {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
};
const auto gc_offsets = opus::layout_to_offsets<T::VEC_C>(u_gc);
auto stage_output_row = [&](auto index_i, int half_m, int half_n,
const AccFragment& c0, const AccFragment& c1,
const AccFragment& c2, const AccFragment& c3) {
constexpr int index = decltype(index_i)::value;
constexpr int n_repeat_stride = T::T_N * T::W_N;
const int row_offset = gc_offsets[index] + c_offset(half_m, half_n);
store<T::VEC_C>(s_c, cast<D_C>(c0), row_offset);
store<T::VEC_C>(s_c, cast<D_C>(c1), row_offset + n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c2), row_offset + 2 * n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c3), row_offset + 3 * n_repeat_stride);
};
auto store_output_bf16 = [&](auto half_m_i, auto half_n_i) {
constexpr int half_m = decltype(half_m_i)::value;
constexpr int half_n = decltype(half_n_i)::value;
__builtin_amdgcn_sched_barrier(0);
if constexpr (!(half_m == 1 && half_n == 0)) {
s_waitcnt_vmcnt(0_I);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
opus::static_for<8>([&](auto copy_i) { copy_output_bf16(half_m, half_n, decltype(copy_i)::value); });
};
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
stage_output_row(opus::number<0>{}, 0, 0, c00[0], c00[1], c00[2], c00[3]);
stage_output_row(opus::number<4>{}, 0, 0, c00[4], c00[5], c00[6], c00[7]);
stage_output_row(opus::number<8>{}, 0, 0, c00[8], c00[9], c00[10], c00[11]);
stage_output_row(opus::number<12>{}, 0, 0, c00[12], c00[13], c00[14], c00[15]);
store_output_bf16(opus::number<0>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 1, 0, c10[0], c10[1], c10[2], c10[3]);
stage_output_row(opus::number<4>{}, 1, 0, c10[4], c10[5], c10[6], c10[7]);
stage_output_row(opus::number<8>{}, 1, 0, c10[8], c10[9], c10[10], c10[11]);
stage_output_row(opus::number<12>{}, 1, 0, c10[12], c10[13], c10[14], c10[15]);
store_output_bf16(opus::number<1>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 0, 1, c01[0], c01[1], c01[2], c01[3]);
stage_output_row(opus::number<4>{}, 0, 1, c01[4], c01[5], c01[6], c01[7]);
stage_output_row(opus::number<8>{}, 0, 1, c01[8], c01[9], c01[10], c01[11]);
stage_output_row(opus::number<12>{}, 0, 1, c01[12], c01[13], c01[14], c01[15]);
store_output_bf16(opus::number<0>{}, opus::number<1>{});
stage_output_row(opus::number<0>{}, 1, 1, c11[0], c11[1], c11[2], c11[3]);
stage_output_row(opus::number<4>{}, 1, 1, c11[4], c11[5], c11[6], c11[7]);
stage_output_row(opus::number<8>{}, 1, 1, c11[8], c11[9], c11[10], c11[11]);
stage_output_row(opus::number<12>{}, 1, 1, c11[12], c11[13], c11[14], c11[15]);
store_output_bf16(opus::number<1>{}, opus::number<1>{});
}
template<class Traits>
__global__ __launch_bounds__(256, 1) void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_body_gfx950<Traits>(kargs);
}

#endif
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_15) && !defined(OPUS_BP_INCLUDE_QUADRANT_15_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_15_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_15 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_14;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0








using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits = opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits_gfx950<true>;
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t byte_limit = 2147483647;
AITER_CHECK(m <= byte_limit / k && n <= byte_limit / k &&
m <= (byte_limit / sizeof(D_C)) / n,
entry, ": tensor byte extent exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ/Y must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = m * n;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 255) / 256;
const dim3 grid(n / 256, tiles_m);
opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits><<<
grid, dim3(256), 0, aiter::getCurrentHIPStream()>>>(args);
}
#else

// SPDX-License-Identifier: Apache-2.0
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)






#endif
#ifdef OPUS_FUSED_HOST_TU


#else

#endif
using opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits = opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits_gfx950<true>;
#if !defined(__HIP_DEVICE_COMPILE__) && !defined(__HIPCC_RTC__)
template <typename D_C>
void opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57(
aiter_tensor_t &XQ, aiter_tensor_t &WQ,
aiter_tensor_t &x_scale, aiter_tensor_t &w_scale, aiter_tensor_t &Y,
std::optional<aiter_tensor_t> workspace)
{
static_assert(std::is_same_v<D_C, bf16_t>);
constexpr const char* entry =
"opus_gemm_a8w8_blockscale_bpreshuffle_launch";
AITER_CHECK((XQ.dim() == 2 || XQ.dim() == 3) &&
WQ.dim() == XQ.dim() && Y.dim() == XQ.dim(),
entry, ": XQ/WQ/Y must have matching rank 2 or 3");
AITER_CHECK(XQ.dim() == 2 ||
(XQ.size(0) == 1 && WQ.size(0) == 1 && Y.size(0) == 1),
entry, ": only batch=1 is supported");
AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous(),
entry, ": XQ/WQ/Y must be contiguous");
AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
entry, ": XQ/WQ must be FP8");
AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
entry, ": output dtype must be bf16");
const int64_t m = XQ.size(-2), n = WQ.size(-2), k = XQ.size(-1);
AITER_CHECK(m > 0 && n > 0 && k > 0 &&
m % 64 == 0 && n % 256 == 0 && k % 128 == 0,
entry, ": requires positive M multiple of 64, N multiple of 256 and K multiple of 128");
AITER_CHECK(WQ.size(-1) == k && Y.size(-2) == m && Y.size(-1) == n,
entry, ": XQ/WQ/Y shapes do not match");
constexpr int64_t byte_limit = 2147483647;
AITER_CHECK(m <= byte_limit / k && n <= byte_limit / k &&
m <= (byte_limit / sizeof(D_C)) / n,
entry, ": tensor byte extent exceeds signed 32-bit addressing");
const auto is_e8m0 = [](const aiter_tensor_t& t) {
return t.dtype() == AITER_DTYPE_fp8_e8m0 || t.dtype() == AITER_DTYPE_u8;
};
AITER_CHECK(is_e8m0(x_scale) && is_e8m0(w_scale),
entry, ": scales must contain one-byte E8M0 values");
AITER_CHECK(x_scale.dim() == 2 && x_scale.size(0) == m && x_scale.size(1) == k / 128,
entry, ": x_scale must be logical [M,K/128]");
AITER_CHECK(x_scale.stride(0) == 1 && x_scale.stride(1) == m,
entry, ": x_scale must have dense column-major storage");
AITER_CHECK(w_scale.dim() == 2 && w_scale.size(0) == n / 128 &&
w_scale.size(1) == k / 128 && w_scale.is_contiguous(),
entry, ": w_scale must be row-major [N/128,K/128]");
const uintptr_t output = reinterpret_cast<uintptr_t>(Y.data_ptr());
const uint64_t output_bytes = uint64_t(m) * n * sizeof(D_C);
for (const auto* input : {&XQ, &WQ, &x_scale, &w_scale}) {
const uintptr_t begin = reinterpret_cast<uintptr_t>(input->data_ptr());
const uint64_t bytes = input->numel() * input->element_size();
const bool overlap = output >= begin
? output - begin < bytes : begin - output < output_bytes;
AITER_CHECK(!overlap, entry, ": Y must not overlap input storage");
}
AITER_CHECK(reinterpret_cast<uintptr_t>(XQ.data_ptr()) % 16 == 0 &&
reinterpret_cast<uintptr_t>(WQ.data_ptr()) % 16 == 0 && output % 16 == 0,
entry, ": XQ/WQ/Y must be 16-byte aligned");
AITER_CHECK(!workspace.has_value(), entry, ": kernel does not use workspace");
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args{};
args.ptr_a = XQ.data_ptr(); args.ptr_b = WQ.data_ptr(); args.ptr_c = Y.data_ptr();
args.m = m; args.n = n; args.k = k; args.batch = 1;
args.stride_a = k; args.stride_b = k; args.stride_c = n;
args.stride_a_batch = m * k; args.stride_b_batch = n * k; args.stride_c_batch = m * n;
args.ptr_sfa = x_scale.data_ptr(); args.ptr_sfb = w_scale.data_ptr();
args.stride_sfa = m; args.stride_sfb = k / 128;
args.stride_sfa_batch = m * (k / 128);
args.stride_sfb_batch = (n / 128) * (k / 128);
const int tiles_m = (m + 255) / 256;
const dim3 grid(n / 256, tiles_m);
opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel<opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57_Traits><<<
grid, dim3(256), 0, aiter::getCurrentHIPStream()>>>(args);
}
#endif

#endif
}
#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_16) && !defined(OPUS_BP_INCLUDE_QUADRANT_16_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_16_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_16 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_15;
#ifdef OPUS_FUSED_HOST_TU
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 0, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M == 1856 && N == 7168 && K == 3072, "requires m_align");
AITER_CHECK(splitK == 1, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}
#else
#if !defined(__HIP_DEVICE_COMPILE__)
// SPDX-License-Identifier: Apache-2.0


static thread_local std::string remote_catalog_error;

int launch_config(
int config_index, int splitK, const void* x, const void* w,
const void* sfa, const void* sfb, void* out, void* workspace,
uint64_t workspace_bytes, int M, int N, int K, void* stream) {
try {
remote_catalog_error.clear();
AITER_CHECK(config_index == 0, "SO/catalog index mismatch");
AITER_CHECK(M>0 && N>0 && K>0, "dimensions must be positive");
AITER_CHECK(x && w && sfa && sfb && out, "input/output pointers must be non-null");
AITER_CHECK(M == 1856 && N == 7168 && K == 3072, "requires m_align");
AITER_CHECK(splitK == 1, "static configuration requires splitK=0");
AITER_CHECK(workspace_bytes <= uint64_t(9223372036854775807), "workspace bytes exceed int64");
AITER_CHECK(workspace_bytes % 4 == 0, "workspace byte capacity must be divisible by four");
auto XQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(x), AITER_DTYPE_fp8, {M,K}, {K,1});
auto WQ = opus_bpreshuffle_host::make_tensor(const_cast<void*>(w), AITER_DTYPE_fp8, {N,K}, {K,1});
auto x_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfa), AITER_DTYPE_fp8_e8m0, {M,K/128}, {1,M});
auto w_scale = opus_bpreshuffle_host::make_tensor(const_cast<void*>(sfb), AITER_DTYPE_fp8_e8m0, {N/128,K/128}, {K/128,1});
auto Y = opus_bpreshuffle_host::make_tensor(out, AITER_DTYPE_bf16, {M,N}, {N,1});
std::optional<aiter_tensor_t> ws;
if(workspace != nullptr) ws = opus_bpreshuffle_host::make_tensor(workspace, AITER_DTYPE_fp32,
std::initializer_list<int64_t>{int64_t(workspace_bytes/4)},
std::initializer_list<int64_t>{1});
else AITER_CHECK(workspace_bytes == 0, "workspace capacity requires pointer");


try { opus_gemm_gfx950_a8w8_mxscale_gemm_bpreshuffle_256x256x256x128_2x2_16x16x128_1x128x128_tiles1_cfg_quadrant_4wave_dbdfb7ecd4569b57<bf16_t>(XQ,WQ,x_scale,w_scale,Y,ws); }
catch(...) {  throw; }

return static_cast<int>(hipGetLastError());
} catch(const std::exception& error) {
remote_catalog_error=error.what();
return -1;
}
}

template<int Selector, int Variant, int Split>
inline int launch(const void* a, const void* b, void* out, const void* sfa, const void* sfb, void* ws, uint64_t bytes, int m, int n, int k, void* stream) {
    return launch_config(Selector, Split, a, b, sfa, sfb, out, ws, bytes, m, n, k, stream);
}

#endif

#endif
}
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template __global__ void opus_bpreshuffle_gfx950::quadrant::schedule_14::opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel<opus_bpreshuffle_gfx950::quadrant::schedule_6::opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_traits_gfx950<true>>(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950);
#endif

#endif

#if defined(OPUS_BP_INCLUDE_QUADRANT_17) && !defined(OPUS_BP_INCLUDE_QUADRANT_17_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_17_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_17 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_6;
#ifdef OPUS_FUSED_HOST_TU

// SPDX-License-Identifier: Apache-2.0



template<class Traits>
__global__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) ;
#else

// SPDX-License-Identifier: Apache-2.0



#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "kid 9000 requires yuyzhang512/llvm-project branch amdgpu-pin-op-dst"
#else
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
return result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
constexpr int threads_k = T::B_K / T::VEC_A;
constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
constexpr auto ga_block_shape = opus::make_tuple(
opus::number<T::HALF_B_M / threads_m_per_block>{},
opus::number<T::T_N>{},
opus::number<threads_m_per_wave>{},
opus::number<T::T_M>{},
opus::number<threads_k>{},
opus::number<T::VEC_A>{});
constexpr auto ga_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_A>(
ga_block_shape,
opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,lane_id % threads_k}));
}
template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
constexpr auto sa_block_shape = opus::make_tuple(
opus::number<T::smem_m_rep / num_waves>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::VEC_A>{});
constexpr auto sa_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sa_block_shape,
opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
constexpr int lanes_n = T::W_N;
constexpr int lanes_k = T::WARP_SIZE / lanes_n;
constexpr int k_per_wave_load = lanes_k * T::VEC_B;
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto gb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto gb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
gb_block_shape,
opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto sb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::VEC_B>{});
constexpr auto sb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sb_block_shape,
opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M / T::T_M>{},
opus::number<T::E_K>{},
opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
constexpr auto rb_block_shape = opus::make_tuple(
opus::number<T::E_N>{},
opus::number<T::T_N>{},
opus::number<T::E_K>{},
opus::number<k_vectors>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto rb_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
rb_block_shape,
opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SCALE_A>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SF>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_M>{},
opus::number<T::W_M>{},
opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<1>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
template<class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((kargs.m | kargs.n) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
const int lane_id = thread_id_x() % T::WARP_SIZE;
auto g_a = [&] {
if constexpr (T::PAD_M) {
const unsigned int a_bytes = static_cast<unsigned int>((kargs.m - row) * kargs.stride_a);
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, a_bytes);
} else {
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a);
}
}();
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
auto g_c = [&] {
if constexpr (T::PAD_M) {
const unsigned int c_bytes = static_cast<unsigned int>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C));
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col, c_bytes);
} else {
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
}
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row, static_cast<unsigned int>(kargs.stride_sfa_batch - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * kargs.stride_sfb, static_cast<unsigned int>(kargs.stride_sfb));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * kargs.stride_a + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * kargs.stride_a + (lane_id % a_k_lanes) * T::VEC_A : wave_id_m * T::HALF_B_N * kargs.stride_b + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * kargs.stride_a;
const int matrix_odd_stride = prefetches_a ? kargs.stride_a : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (T::PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = static_cast<unsigned int>(kargs.k) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (T::PAD_M) {
sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < kargs.m) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
} else {
if (first_k_group < scale_k_groups) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = static_cast<unsigned int>(kargs.k) / T::B_K;
if constexpr (T::PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma unroll 2
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 3>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
mma_scale_group<T, 6, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
int epilogue_lane_id = lane_id;
asm volatile("" : "+v"(epilogue_lane_id));
auto p_coord_c = opus::make_tuple(wave_id_m, epilogue_lane_id % mma.grpn_c, wave_id_n, epilogue_lane_id / mma.grpn_c);
auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(c_lds_row_stride_elems, 1_I), p_coord_c);
auto c_offset = [&](int half_tile_m, int half_tile_n) { return half_tile_m * T::HALF_B_M * c_lds_row_stride_elems + half_tile_n * T::HALF_B_N; };
const int output_thread_id = wave_id * T::WARP_SIZE + epilogue_lane_id;
auto copy_output_bf16 = [&](int half_m, int half_n, int copy_index) {
const int linear = output_thread_id * 8 + copy_index * T::BLOCK_SIZE * 8;
const int output_row = linear / T::HALF_B_N + half_m * T::HALF_B_M;
const int output_col = linear % T::HALF_B_N + half_n * T::HALF_B_N;
const auto value = load<8>(s_c, output_row * c_lds_row_stride_elems + output_col);
if constexpr (T::PAD_M) {
if (row + output_row < kargs.m) {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
} else {
store<8>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
}
};
const auto gc_offsets = opus::layout_to_offsets<T::VEC_C>(u_gc);
auto stage_output_row = [&](auto index_i, int half_m, int half_n,
const AccFragment& c0, const AccFragment& c1,
const AccFragment& c2, const AccFragment& c3) {
constexpr int index = decltype(index_i)::value;
constexpr int n_repeat_stride = T::T_N * T::W_N;
const int row_offset = gc_offsets[index] + c_offset(half_m, half_n);
store<T::VEC_C>(s_c, cast<D_C>(c0), row_offset);
store<T::VEC_C>(s_c, cast<D_C>(c1), row_offset + n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c2), row_offset + 2 * n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c3), row_offset + 3 * n_repeat_stride);
};
auto store_output_bf16 = [&](auto half_m_i, auto half_n_i) {
constexpr int half_m = decltype(half_m_i)::value;
constexpr int half_n = decltype(half_n_i)::value;
__builtin_amdgcn_sched_barrier(0);
if constexpr (!(half_m == 1 && half_n == 0)) {
s_waitcnt_vmcnt(0_I);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
opus::static_for<8>([&](auto copy_i) { copy_output_bf16(half_m, half_n, decltype(copy_i)::value); });
};
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
stage_output_row(opus::number<0>{}, 0, 0, c00[0], c00[1], c00[2], c00[3]);
stage_output_row(opus::number<4>{}, 0, 0, c00[4], c00[5], c00[6], c00[7]);
stage_output_row(opus::number<8>{}, 0, 0, c00[8], c00[9], c00[10], c00[11]);
stage_output_row(opus::number<12>{}, 0, 0, c00[12], c00[13], c00[14], c00[15]);
store_output_bf16(opus::number<0>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 1, 0, c10[0], c10[1], c10[2], c10[3]);
stage_output_row(opus::number<4>{}, 1, 0, c10[4], c10[5], c10[6], c10[7]);
stage_output_row(opus::number<8>{}, 1, 0, c10[8], c10[9], c10[10], c10[11]);
stage_output_row(opus::number<12>{}, 1, 0, c10[12], c10[13], c10[14], c10[15]);
store_output_bf16(opus::number<1>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 0, 1, c01[0], c01[1], c01[2], c01[3]);
stage_output_row(opus::number<4>{}, 0, 1, c01[4], c01[5], c01[6], c01[7]);
stage_output_row(opus::number<8>{}, 0, 1, c01[8], c01[9], c01[10], c01[11]);
stage_output_row(opus::number<12>{}, 0, 1, c01[12], c01[13], c01[14], c01[15]);
store_output_bf16(opus::number<0>{}, opus::number<1>{});
stage_output_row(opus::number<0>{}, 1, 1, c11[0], c11[1], c11[2], c11[3]);
stage_output_row(opus::number<4>{}, 1, 1, c11[4], c11[5], c11[6], c11[7]);
stage_output_row(opus::number<8>{}, 1, 1, c11[8], c11[9], c11[10], c11[11]);
stage_output_row(opus::number<12>{}, 1, 1, c11[12], c11[13], c11[14], c11[15]);
store_output_bf16(opus::number<1>{}, opus::number<1>{});
}
template<class Traits>
__global__ __launch_bounds__(256, 1)
void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_kernel(
opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_body_gfx950<Traits>(kargs);
}
#endif
#endif

#endif
}
#endif


#if defined(OPUS_BP_INCLUDE_QUADRANT_18) && !defined(OPUS_BP_INCLUDE_QUADRANT_18_DONE)
#define OPUS_BP_INCLUDE_QUADRANT_18_DONE
namespace opus_bpreshuffle_gfx950::quadrant::schedule_18 {
using namespace opus_bpreshuffle_gfx950::quadrant::schedule_0;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
#if !__has_cpp_attribute(clang::amdgpu_pin_agpr)
#error "kid 9000 requires yuyzhang512/llvm-project branch amdgpu-pin-op-dst"
#else
using opus::operator""_I;
constexpr int A0_AGPR_BASE = 0;
constexpr int A1_AGPR_BASE = 32;
constexpr int B0_AGPR_BASE = 64;
constexpr int B1_AGPR_BASE = 96;
constexpr int A0_M0_STAGE_AGPR_BASE = 128;
constexpr int C11_TAIL_AGPR_BASE = 128;
constexpr int C01_AGPR_BASE = 148;
template<class T, int N, int AGPR_BASE>
struct pinned_accumulator_array : opus::array<T, N> {};
template<class T>
struct accumulator_pin_traits {
static constexpr bool enabled = false;
static constexpr int agpr_base = 0;
};
template<class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
static constexpr bool enabled = true;
static constexpr int agpr_base = AGPR_BASE;
};
template<int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment& fragment, Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
constexpr int physical_reg = decltype(reg)::value;
static_assert(physical_reg >= 0 && physical_reg < 128);
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]]
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}
template<int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem& smem, const Offsets& offsets) {
using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
opus::array<Chunk, 2> values;
opus::static_for<2>([&](auto i) {
constexpr int chunk = decltype(i)::value;
values[chunk] = opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
});
Fragment result;
opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
return result;
}
template<int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk* chunks, Mem& smem, const Layout& layout) {
constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
static_assert(AGPR_BASE >= 0);
static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
const auto offsets = opus::layout_to_offsets<Vec>(layout);
opus::static_for<CHUNK_COUNT>([&](auto i) {
constexpr int chunk = decltype(i)::value;
[[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]]
chunks[chunk] = opus::load<Vec>(smem, offsets[chunk]);
});
}
template<class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0, class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA& mma, const VA& v_a, const VB& v_b, VC& v_c, unsigned int v_sfa, unsigned int v_sfb) {
using tiled_mma = opus::remove_cvref_t<MMA>;
using base_mma = typename tiled_mma::MMA;
opus::static_for<COUNT>([&](auto i) {
constexpr int c_index = C_BEGIN + decltype(i)::value;
constexpr int m_repeat = c_index / T::E_N;
constexpr int n_repeat = c_index % T::E_N;
using pin_traits = accumulator_pin_traits<opus::remove_cvref_t<VC>>;
constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
if constexpr (EXPLICIT_AGPR_BASE >= 0) {
[[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE + (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else if constexpr (pin_traits::enabled) {
[[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]]
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
} else {
v_c[c_index] = base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa), static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
}
});
}
template<class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
constexpr int threads_k = T::B_K / T::VEC_A;
constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
constexpr auto ga_block_shape = opus::make_tuple(
opus::number<T::HALF_B_M / threads_m_per_block>{},
opus::number<T::T_N>{},
opus::number<threads_m_per_wave>{},
opus::number<T::T_M>{},
opus::number<threads_k>{},
opus::number<T::VEC_A>{});
constexpr auto ga_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_A>(
ga_block_shape,
opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,lane_id % threads_k}));
}
template<class T>
__device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
constexpr auto sa_block_shape = opus::make_tuple(
opus::number<T::smem_m_rep / num_waves>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::VEC_A>{});
constexpr auto sa_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sa_block_shape,
opus::unfold_x_stride(sa_block_dim, sa_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
constexpr int lanes_n = T::W_N;
constexpr int lanes_k = T::WARP_SIZE / lanes_n;
constexpr int k_per_wave_load = lanes_k * T::VEC_B;
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto gb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto gb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
gb_block_shape,
opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
constexpr int k_repeats = T::B_K / k_per_wave_load;
constexpr auto sb_block_shape = opus::make_tuple(
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<n_repeats>{},
opus::number<k_repeats>{},
opus::number<T::VEC_B>{});
constexpr auto sb_block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout(
sb_block_shape,
opus::unfold_x_stride(sb_block_dim, sb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
constexpr auto ra_block_shape = opus::make_tuple(
opus::number<T::E_M>{},
opus::number<T::T_M>{},
opus::number<T::T_M>{},
opus::number<T::W_M / T::T_M>{},
opus::number<T::E_K>{},
opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
opus::number<T::WARP_SIZE / T::W_M>{},
opus::number<T::VEC_A>{});
constexpr auto ra_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = lane_id % T::W_M;
return opus::make_layout<T::VEC_A>(
ra_block_shape,
opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, lane_id_m / T::T_M, lane_id / T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
constexpr auto rb_block_shape = opus::make_tuple(
opus::number<T::E_N>{},
opus::number<T::T_N>{},
opus::number<T::E_K>{},
opus::number<k_vectors>{},
opus::number<T::WARP_SIZE>{},
opus::number<T::VEC_B>{});
constexpr auto rb_block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<T::VEC_B>(
rb_block_shape,
opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SCALE_A>{},
opus::number<T::VEC_SCALE_A>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
return opus::make_layout<T::VEC_SCALE_A>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{},
opus::number<T::T_N>{},
opus::number<T::T_M>{},
opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
opus::number<T::B_M / T::VEC_SF>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(
opus::number<T::T_M>{},
opus::number<T::W_M>{},
opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}
template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<1>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
return opus::make_layout<1>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
constexpr auto block_shape = opus::make_tuple(
opus::number<T::SCALE_N_HALVES>{},
opus::number<T::SCALE_PANEL_K_CAPACITY>{},
opus::number<T::VEC_SF>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::p_dim{}),
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<T::VEC_SF>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::VEC_SF>{}, opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}
template<class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
constexpr auto block_dim = opus::make_tuple(
opus::make_tuple(opus::y_dim{}));
return opus::make_layout<Vec>(
block_shape,
opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
opus::unfold_p_coord(block_dim, opus::tuple{}));
}
template<class Traits>
__device__ __forceinline__ void opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_body_gfx950(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 kargs) {
using namespace opus;
using T = opus::remove_cvref_t<Traits>;
using D_A = opus::fp8_t;
using D_B = opus::fp8_t;
using D_C = opus::bf16_t;
using D_ACC = opus::fp32_t;
using D_SF = unsigned char;
using D_SF_PACK = unsigned int;
int block_m = block_id_y();
int block_n = block_id_x();
if (((65536 | 16384) & 511) == 0) {
const int old_m = block_m;
block_m = (block_m & ~1) | (block_n & 1);
block_n = (block_n & ~1) | (old_m & 1);
}
const int row = block_m * T::B_M;
const int col = block_n * T::B_N;
const int batch_id = block_id_z();
const unsigned tid=thread_id_x();
__builtin_assume(tid<T::BLOCK_SIZE);
const unsigned wave_id=__builtin_amdgcn_readfirstlane(tid/T::WARP_SIZE);
const unsigned lane_id=tid%T::WARP_SIZE;
auto g_a = [&] {
if constexpr (T::PAD_M) {
const unsigned int a_bytes = static_cast<unsigned int>((65536 - row) * 1536);
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * 1536, a_bytes);
} else {
return make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * 1536);
}
}();
auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * 1536);
auto g_c = [&] {
if constexpr (T::PAD_M) {
const unsigned int c_bytes = static_cast<unsigned int>(((65536 - row) * 16384 - col) * sizeof(D_C));
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * 16384 + col, c_bytes);
} else {
return make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * 16384 + col);
}
}();
auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row, static_cast<unsigned int>(kargs.stride_sfa_batch - row));
auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N + wave_id % T::SCALE_N_HALVES) * 12, static_cast<unsigned int>(12));
const int wave_id_m = wave_id % T::T_M;
const int wave_id_n = wave_id / T::T_M;
const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, 1536);
const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, 1536);
const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
const auto u_gsfa = make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, 65536);
const auto u_rsfa = make_layout_rsfa_scale<T>(lane_id, wave_id_m);
const auto u_rsfa_pair = make_layout_rsfa_scale<T, T::VEC_SCALE_SF_PAIR>(lane_id, wave_id_m);
const auto u_gsfb = make_layout_gsfb_scale<T>(lane_id, wave_id);
const auto u_ssfb = make_layout_ssfb_scale<T>(lane_id, wave_id);
const auto u_rsfb = make_layout_rsfb_scale<T>();
const auto u_rsfb_pair = make_layout_rsfb_scale<T, T::VEC_SCALE_SF_PAIR>();
constexpr int smem_a_elem = T::smem_m_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int smem_b_elem = T::smem_n_rep * (T::smem_linear_wave + T::smem_padding);
constexpr int matrix_lds_bytes = smem_a_elem * 4 * sizeof(D_A) + smem_b_elem * 4 * sizeof(D_B);
alignas(16) __shared__ char smem_matrix[matrix_lds_bytes];
auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + smem_a_elem * 4 * sizeof(D_A)));
auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
alignas(8) __shared__ char smem_sfa[T::SFA_PANEL_BYTES];
alignas(8) __shared__ char smem_sfb[T::SFB_PANEL_BYTES];
auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_sfa));
auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_sfb));
auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
seq<T::E_M, T::E_N, T::E_K>{},
seq<T::T_M, T::T_N, T::T_K>{},
seq<T::W_M, T::W_N, T::W_K>{},
mfma_adaptor_swap_ab{});
using BaseMMA = typename decltype(mma)::MMA;
using AFragment = typename BaseMMA::vtype_a;
using BFragment = typename BaseMMA::vtype_b;
using AccFragment = typename BaseMMA::vtype_c;
using AChunk = opus::vector_t<D_A, T::VEC_A>;
using BChunk = opus::vector_t<D_B, T::VEC_B>;
constexpr int A_CHUNKS_PER_FRAGMENT = sizeof(AFragment) / sizeof(AChunk);
constexpr int B_CHUNKS_PER_FRAGMENT = sizeof(BFragment) / sizeof(BChunk);
constexpr int A_DWORDS_PER_CHUNK = sizeof(AChunk) / sizeof(opus::u32_t);
opus::array<AFragment, T::E_M> v_a[2];
opus::array<BFragment, T::E_N> v_b;
opus::array<BFragment, T::E_N> v_b_second;
auto* a0_chunks = reinterpret_cast<AChunk*>(&v_a[0]);
auto* a1_chunks = reinterpret_cast<AChunk*>(&v_a[1]);
auto* b0_chunks = reinterpret_cast<BChunk*>(&v_b);
auto* b1_chunks = reinterpret_cast<BChunk*>(&v_b_second);
using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
using PinnedC01Tile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, C01_AGPR_BASE>;
CTile c00{};
PinnedC01Tile c01{};
CTile c10{};
CTile c11{};
constexpr int c_lds_row_stride_elems = T::B_N + 8;
D_SF_PACK v_sfa[2];
D_SF_PACK v_sfb[T::SCALE_N_HALVES];
opus::vector_t<D_SF_PACK, 2> v_sfb_next;
opus::vector_t<D_SF_PACK, 2> v_sfa_next;
constexpr int scale_panel_mask = T::SCALE_PANEL_K_CAPACITY - 1;
auto ga_offset = [&](int half_tile_m, int tile_k) { return half_tile_m * T::HALF_B_M * 1536 + tile_k * T::B_K; };
auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * 1536 + tile_k * T::B_K * 16; };
auto sa_offset = [&](int stage, int half_tile_m) { return (stage * 2 + half_tile_m) * smem_a_elem; };
auto sb_offset = [&](int stage, int half_tile_n) { return (stage * 2 + half_tile_n) * smem_b_elem; };
auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * 65536; };
auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
auto ssfa_offset = [&](int k_tile, int half_tile_m) { return (k_tile & scale_panel_mask) * T::SFA_PANEL_PITCH + half_tile_m * T::VEC_SCALE_SF; };
auto ssfb_offset = [&](int k_tile, int half_tile_n) { return (k_tile & scale_panel_mask) * T::SCALE_N_HALVES * T::VEC_SCALE_SF + half_tile_n * T::VEC_SCALE_SF; };
const bool prefetches_a = wave_id_n == 0;
auto g_matrix_prefetch = prefetches_a ? g_a : g_b;
constexpr int a_k_lanes = T::B_K / T::VEC_A;
constexpr int matrix_lds_pitch = T::smem_linear_wave + T::smem_padding;
const int matrix_lane_offset = prefetches_a ? (wave_id_m * T::HALF_B_M + (lane_id / a_k_lanes) * 2) * 1536 + (lane_id % a_k_lanes) * T::VEC_A : wave_id_m * T::HALF_B_N * 1536 + lane_id * T::VEC_B;
const int matrix_pair_stride = 16 * 1536;
const int matrix_odd_stride = prefetches_a ? 1536 : T::WARP_SIZE * T::VEC_B;
const int matrix_tile_stride = prefetches_a ? T::B_K : T::B_K * T::VEC_B;
auto* matrix_lds_base = (prefetches_a ? s_a.ptr : s_b.ptr) + wave_id_m * smem_a_elem;
opus::vector_t<int, 4> matrix_voffsets;
opus::static_for<4>([&](auto local_i) {
constexpr int local = decltype(local_i)::value;
constexpr int immediate = local * matrix_lds_pitch;
int voffset = matrix_lane_offset + (local / 2) * matrix_pair_stride + (local % 2) * matrix_odd_stride - immediate;
asm volatile("" : "+v"(voffset));
matrix_voffsets[local] = voffset;
});
auto issue_matrix_prefetch = [&](auto issue_i, int matrix_stage, int tile_offset) {
constexpr int issue = decltype(issue_i)::value;
constexpr int local = issue % 4;
constexpr int group = issue / 4;
constexpr int immediate = local * matrix_lds_pitch;
auto* dst = matrix_lds_base + matrix_stage * 2 * smem_a_elem + group * 4 * matrix_lds_pitch;
if constexpr (T::PAD_M) {
int voffset = static_cast<int>(matrix_voffsets[local]) + __builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride);
asm volatile("" : "+v"(voffset));
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
voffset, 0, opus::number<immediate>{}, opus::number<0>{});
} else {
async_load<16>(
g_matrix_prefetch,
reinterpret_cast<void*>(reinterpret_cast<__UINTPTR_TYPE__>(dst)),
static_cast<int>(matrix_voffsets[local]),
__builtin_amdgcn_readfirstlane(tile_offset + group * 2 * matrix_pair_stride),
opus::number<immediate>{},
opus::number<0>{});
}
};
const int scale_k_groups = static_cast<unsigned int>(1536) / T::GROUP_K;
constexpr int sfa_lds_word_stride = T::VEC_SF * (T::B_M / T::HALF_B_M) * T::E_M;
const auto sfa_gmem_offsets = opus::layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
opus::vector_t<D_SF, T::VEC_SCALE_A> sfa_panel_raw[T::SFA_PASSES_PER_CACHE_PANEL];
auto load_sfa_panel = [&](int panel_k_begin) {
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if constexpr (T::PAD_M) {
sfa_panel_raw[pass] = {};
if (first_k_group < scale_k_groups) {
const int scale_row = ((lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2)) * T::VEC_SCALE_A;
if (row + scale_row < 65536) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
} else {
if (first_k_group < scale_k_groups) {
sfa_panel_raw[pass] = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
}
}
});
};
auto publish_sfa_panel = [&](int panel_k_begin) {
int publish_lane_id = lane_id;
asm volatile("" : "+v"(publish_lane_id));
const auto publish_u_ssfa = make_layout_ssfa_scale<T>(publish_lane_id, wave_id_m, wave_id_n);
const auto publish_sfa_smem_offsets = opus::layout_to_offsets<T::VEC_SF>(publish_u_ssfa);
const int lane_in_quad = publish_lane_id % 4;
const unsigned int pair_byte_select = (lane_in_quad % 2) ? 0x03070105u : 0x06020400u;
const unsigned int quad_byte_select = (lane_in_quad / 2) ? 0x03020706u : 0x05040100u;
auto transpose_sfa_word_4x4 = [&](D_SF_PACK lane_word) {
const D_SF_PACK adjacent_lane = opus::mov_dpp(lane_word, opus::number<0xb1>{});
const D_SF_PACK paired_bytes = __builtin_amdgcn_perm(adjacent_lane, lane_word, pair_byte_select);
const D_SF_PACK opposite_pair = opus::mov_dpp(paired_bytes, opus::number<0x4e>{});
return __builtin_amdgcn_perm(opposite_pair, paired_bytes, quad_byte_select);
};
opus::static_for<T::SFA_PASSES_PER_CACHE_PANEL>([&](auto pass_i) {
constexpr int pass = decltype(pass_i)::value;
const int first_k_group = panel_k_begin + pass * T::SFA_K_COLUMNS_PER_PASS + wave_id * T::SFA_K_COLUMNS_PER_WAVE;
if (first_k_group < scale_k_groups) {
const auto raw_words = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 4>, sfa_panel_raw[pass]);
opus::static_for<4>([&](auto word_i) {
constexpr int word = decltype(word_i)::value;
const D_SF_PACK packed_scales = transpose_sfa_word_4x4(raw_words[word]);
store<T::VEC_SF>(s_sfa, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed_scales), publish_sfa_smem_offsets[pass] + word * sfa_lds_word_stride);
});
}
});
};
auto refill_scale_panel = [&](int next_tile) {
if ((next_tile & scale_panel_mask) == 0) {
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
load_sfa_panel(next_tile);
D_SF_PACK raw_b = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(next_tile));
raw_b = static_cast<D_SF_PACK>(raw[0]);
}
s_waitcnt_vmcnt(0_I);
publish_sfa_panel(next_tile);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = raw_b;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
}
};
int stage = 0;
int tile = 0;
const int loops = static_cast<unsigned int>(1536) / T::B_K;
if constexpr (T::PAD_M) {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(0, 0), u_sa + sa_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga + ga_offset(1, 0), u_sa + sa_offset(0, 1));
} else {
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 0), ga_offset(0, 0));
async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(0, 1), ga_offset(1, 0));
}
load_sfa_panel(0);
D_SF_PACK panel_sfb_raw = 0;
if (wave_id < T::SCALE_N_HALVES) {
const auto raw = load<1>(g_sfb, u_gsfb + gsfb_offset(0));
panel_sfb_raw = static_cast<D_SF_PACK>(raw[0]);
}
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 0), gb_offset(0, 0));
async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(0, 1), gb_offset(1, 0));
publish_sfa_panel(0);
if (wave_id < T::SCALE_N_HALVES) {
const D_SF_PACK packed = panel_sfb_raw;
store<T::VEC_SF>(s_sfb, __builtin_bit_cast(opus::vector_t<D_SF, T::VEC_SF>, packed), u_ssfb);
}
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
if (loops > 1) { opus::static_for<8>([&](auto initial_issue) { issue_matrix_prefetch(initial_issue, 1, matrix_tile_stride); });}
v_sfa[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 0)));
v_sfa[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfa, u_rsfa + ssfa_offset(0, 1)));
v_sfb[0] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 0)));
v_sfb[1] = __builtin_bit_cast(D_SF_PACK, load<T::VEC_SCALE_SF>(s_sfb, u_rsfb + ssfb_offset(0, 1)));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B0_AGPR_BASE>(b0_chunks, s_b, u_rb + sb_offset(stage, 0));
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A0_AGPR_BASE>(a0_chunks, s_a, u_ra + sa_offset(stage, 0));
if (loops > 1) {
opus::static_for<8>([&](auto initial_issue) {
using Issue = opus::number<decltype(initial_issue)::value + 8>;
issue_matrix_prefetch(Issue{}, 1, matrix_tile_stride);
});
}
load_operand_chunks_pinned<T::VEC_A, T::E_M * A_CHUNKS_PER_FRAGMENT, A1_AGPR_BASE>(a1_chunks, s_a, u_ra + sa_offset(stage, 1));
load_operand_chunks_pinned<T::VEC_B, T::E_N * B_CHUNKS_PER_FRAGMENT, B1_AGPR_BASE>(b1_chunks, s_b, u_rb + sb_offset(stage, 1));
__builtin_amdgcn_s_setprio(1);
#pragma unroll 2
for (; tile + 2 < loops; ++tile) {
const int k_tile = tile;
asm volatile("" : "+v"(matrix_voffsets[0]), "+v"(matrix_voffsets[1]), "+v"(matrix_voffsets[2]), "+v"(matrix_voffsets[3]));
refill_scale_panel(k_tile + 1);
const int next_stage = stage ^ 1;
const int future_tile = k_tile + 2;
int future_matrix_offset = __builtin_amdgcn_readfirstlane(future_tile * matrix_tile_stride);
asm volatile("" : "+s"(future_matrix_offset));
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(k_tile + 1, 0)));
mma_scale_group<T, 2, 3>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_barrier(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_s_barrier();
mma_scale_group<T, 6, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 7, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<0>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 8, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<1>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 9, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<2>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 11, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<3>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
mma_scale_group<T, 13, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
issue_matrix_prefetch(opus::number<4>{}, stage, future_matrix_offset);
mma_scale_group<T, 15, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 0);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<5>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 1, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<6>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 3, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(k_tile + 1, 0)));
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<7>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<8>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 7, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<9>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<10>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
issue_matrix_prefetch(opus::number<11>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 1);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 3, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<12>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 1, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 3, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<0>{}, v_a[0][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
issue_matrix_prefetch(opus::number<13>{}, stage, future_matrix_offset);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
issue_matrix_prefetch(opus::number<14>{}, stage, future_matrix_offset);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
issue_matrix_prefetch(opus::number<15>{}, stage, future_matrix_offset);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x800, 1, 2);
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
const BFragment b1_n3_next = load_operand_fragment_staged<T::VEC_B, 6, BFragment>(
s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
mma_scale_group<T, 5, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
const AFragment a1_m3_next = load_operand_fragment_staged<T::VEC_A, 6, AFragment>(
s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 3, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1, C11_TAIL_AGPR_BASE, 11>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
[[clang::amdgpu_pin_agpr(A1_AGPR_BASE + 3 * sizeof(AFragment) / sizeof(opus::u32_t))]] v_a[1][3] = a1_m3_next;
[[clang::amdgpu_pin_agpr(B1_AGPR_BASE + 3 * sizeof(BFragment) / sizeof(opus::u32_t))]] v_b_second[3] = b1_n3_next;
__builtin_amdgcn_sched_barrier(0x004);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
if (loops > 1) {
refill_scale_panel(tile + 1);
const int next_stage = stage ^ 1;
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
v_sfa_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfa, u_rsfa_pair + ssfa_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 0);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 3,0);
__builtin_amdgcn_s_setprio(0);
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
v_sfb_next = __builtin_bit_cast(opus::vector_t<D_SF_PACK, 2>, load<T::VEC_SCALE_SF_PAIR>(s_sfb, u_rsfb_pair + ssfb_offset(tile + 1, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 1, 1);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
AChunk a0_m0_next_0;
AChunk a0_m0_next_1;
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE)]] a0_m0_next_0 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[0]);
[[clang::amdgpu_pin_agpr(A0_M0_STAGE_AGPR_BASE + A_DWORDS_PER_CHUNK)]] a0_m0_next_1 = load<T::VEC_A>(s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0))[1]);
__builtin_amdgcn_sched_group_barrier(0x008, 10, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<64>{}, v_b[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 1);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 1);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<72>{}, v_b[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<80>{}, v_b[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
AFragment a0_m0_joined;
opus::set_slice(a0_m0_joined, a0_m0_next_0, opus::number<0>{}, opus::number<16>{});
opus::set_slice(a0_m0_joined, a0_m0_next_1, opus::number<16>{}, opus::number<32>{});
[[clang::amdgpu_pin_agpr(A0_AGPR_BASE)]] v_a[0][0] = a0_m0_joined;
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 4, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 5, 1>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<88>{}, v_b[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
v_sfb[0] = v_sfb_next[0];
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<8>{}, v_a[0][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<16>{}, v_a[0][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
v_sfa[0] = v_sfa_next[0];
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<24>{}, v_a[0][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 0)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 2);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 2);
__builtin_amdgcn_sched_barrier(0);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 0>(opus::number<32>{}, v_a[1][0], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 2>(opus::number<40>{}, v_a[1][1], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 4, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 8, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 0>(opus::number<96>{}, v_b_second[0], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 9, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 13, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 2>(opus::number<104>{}, v_b_second[1], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 10, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_B, 4>(opus::number<112>{}, v_b_second[2], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
mma_scale_group<T, 11, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 4>(opus::number<48>{}, v_a[1][2], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
mma_scale_group<T, 15, 1>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
load_operand_fragment_pinned<T::VEC_A, 6>(opus::number<56>{}, v_a[1][3], s_a, opus::layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage, 1)));
load_operand_fragment_pinned<T::VEC_B, 6>(opus::number<120>{}, v_b_second[3], s_b, opus::layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage, 1)));
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 2, 3);
__builtin_amdgcn_sched_group_barrier(0x008, 1, 3);
__builtin_amdgcn_sched_group_barrier(0x100, 4, 3);
__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_setprio(0);
v_sfb[1] = v_sfb_next[1];
v_sfa[1] = v_sfa_next[1];
stage = next_stage;
}
__builtin_amdgcn_s_setprio(1);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b, c00, v_sfa[0], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b, c10, v_sfa[1], v_sfb[0]);
mma_scale_group<T, 0, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[0], v_b_second, c01, v_sfa[0], v_sfb[1]);
mma_scale_group<T, 0, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 2, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 4, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 6, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 8, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 10, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 12, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
mma_scale_group<T, 14, 2>(mma, v_a[1], v_b_second, c11, v_sfa[1], v_sfb[1]);
__builtin_amdgcn_s_setprio(0);
int epilogue_lane_id = lane_id;
asm volatile("" : "+v"(epilogue_lane_id));
auto p_coord_c = opus::make_tuple(wave_id_m, epilogue_lane_id % mma.grpn_c, wave_id_n, epilogue_lane_id / mma.grpn_c);
auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(c_lds_row_stride_elems, 1_I), p_coord_c);
auto c_offset = [&](int half_tile_m, int half_tile_n) { return half_tile_m * T::HALF_B_M * c_lds_row_stride_elems + half_tile_n * T::HALF_B_N; };
const int output_thread_id = wave_id * T::WARP_SIZE + epilogue_lane_id;
auto copy_output_bf16 = [&](int half_m, int half_n, int copy_index) {
const int linear = output_thread_id * 8 + copy_index * T::BLOCK_SIZE * 8;
const int output_row = linear / T::HALF_B_N + half_m * T::HALF_B_M;
const int output_col = linear % T::HALF_B_N + half_n * T::HALF_B_N;
const auto value = load<8>(s_c, output_row * c_lds_row_stride_elems + output_col);
if constexpr (T::PAD_M) {
if (row + output_row < 65536) {
store<8>(g_c, value, output_row * 16384 + output_col, 0, opus::number<2>{});
}
} else {
store<8>(g_c, value, output_row * 16384 + output_col, 0, opus::number<2>{});
}
};
const auto gc_offsets = opus::layout_to_offsets<T::VEC_C>(u_gc);
auto stage_output_row = [&](auto index_i, int half_m, int half_n,
const AccFragment& c0, const AccFragment& c1,
const AccFragment& c2, const AccFragment& c3) {
constexpr int index = decltype(index_i)::value;
constexpr int n_repeat_stride = T::T_N * T::W_N;
const int row_offset = gc_offsets[index] + c_offset(half_m, half_n);
store<T::VEC_C>(s_c, cast<D_C>(c0), row_offset);
store<T::VEC_C>(s_c, cast<D_C>(c1), row_offset + n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c2), row_offset + 2 * n_repeat_stride);
store<T::VEC_C>(s_c, cast<D_C>(c3), row_offset + 3 * n_repeat_stride);
};
auto store_output_bf16 = [&](auto half_m_i, auto half_n_i) {
constexpr int half_m = decltype(half_m_i)::value;
constexpr int half_n = decltype(half_n_i)::value;
__builtin_amdgcn_sched_barrier(0);
if constexpr (!(half_m == 1 && half_n == 0)) {
s_waitcnt_vmcnt(0_I);
}
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
opus::static_for<8>([&](auto copy_i) { copy_output_bf16(half_m, half_n, decltype(copy_i)::value); });
};
s_waitcnt_vmcnt(0_I);
s_waitcnt_lgkmcnt(0_I);
__builtin_amdgcn_s_barrier();
__builtin_amdgcn_sched_barrier(0);
stage_output_row(opus::number<0>{}, 0, 0, c00[0], c00[1], c00[2], c00[3]);
stage_output_row(opus::number<4>{}, 0, 0, c00[4], c00[5], c00[6], c00[7]);
stage_output_row(opus::number<8>{}, 0, 0, c00[8], c00[9], c00[10], c00[11]);
stage_output_row(opus::number<12>{}, 0, 0, c00[12], c00[13], c00[14], c00[15]);
store_output_bf16(opus::number<0>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 1, 0, c10[0], c10[1], c10[2], c10[3]);
stage_output_row(opus::number<4>{}, 1, 0, c10[4], c10[5], c10[6], c10[7]);
stage_output_row(opus::number<8>{}, 1, 0, c10[8], c10[9], c10[10], c10[11]);
stage_output_row(opus::number<12>{}, 1, 0, c10[12], c10[13], c10[14], c10[15]);
store_output_bf16(opus::number<1>{}, opus::number<0>{});
stage_output_row(opus::number<0>{}, 0, 1, c01[0], c01[1], c01[2], c01[3]);
stage_output_row(opus::number<4>{}, 0, 1, c01[4], c01[5], c01[6], c01[7]);
stage_output_row(opus::number<8>{}, 0, 1, c01[8], c01[9], c01[10], c01[11]);
stage_output_row(opus::number<12>{}, 0, 1, c01[12], c01[13], c01[14], c01[15]);
store_output_bf16(opus::number<0>{}, opus::number<1>{});
stage_output_row(opus::number<0>{}, 1, 1, c11[0], c11[1], c11[2], c11[3]);
stage_output_row(opus::number<4>{}, 1, 1, c11[4], c11[5], c11[6], c11[7]);
stage_output_row(opus::number<8>{}, 1, 1, c11[8], c11[9], c11[10], c11[11]);
stage_output_row(opus::number<12>{}, 1, 1, c11[12], c11[13], c11[14], c11[15]);
store_output_bf16(opus::number<1>{}, opus::number<1>{});
}

#endif
#endif
}
#endif
