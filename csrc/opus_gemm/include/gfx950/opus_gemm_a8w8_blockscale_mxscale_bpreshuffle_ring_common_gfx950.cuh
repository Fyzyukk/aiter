// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <climits>
#include <type_traits>
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_traits_a8w8_scale_gfx950.cuh"
#include "opus_gemm_kargs_a8w8_mxscale_bpreshuffle_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__) && defined(OPUS_BP_FIXED_M) &&          \
    defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
template <class Arguments>
__device__ __forceinline__ void
opus_gemm_blockscale_mxscale_bpreshuffle_apply_fixed_args(Arguments &a) {
    if constexpr (requires { a.m; })
        a.m = OPUS_BP_FIXED_M;
    if constexpr (requires { a.n; })
        a.n = OPUS_BP_FIXED_N;
    if constexpr (requires { a.k; })
        a.k = OPUS_BP_FIXED_K;
    if constexpr (requires { a.batch; })
        a.batch = 1;
    if constexpr (requires { a.stride_a; })
        a.stride_a = OPUS_BP_FIXED_K;
    if constexpr (requires { a.stride_b; })
        a.stride_b = OPUS_BP_FIXED_K;
    if constexpr (requires { a.stride_c; })
        a.stride_c = OPUS_BP_FIXED_N;
    if constexpr (requires { a.stride_a_batch; })
        a.stride_a_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_M) * OPUS_BP_FIXED_K);
    if constexpr (requires { a.stride_b_batch; })
        a.stride_b_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_N) * OPUS_BP_FIXED_K);
    if constexpr (requires { a.stride_c_batch; })
        a.stride_c_batch = static_cast<int>(int64_t(OPUS_BP_FIXED_M) * OPUS_BP_FIXED_N);
    if constexpr (requires { a.stride_sfa; })
        a.stride_sfa = OPUS_BP_FIXED_M;
    if constexpr (requires { a.stride_sfb; })
        a.stride_sfb = OPUS_BP_FIXED_K / 128;
    if constexpr (requires { a.stride_sfa_batch; })
        a.stride_sfa_batch = OPUS_BP_FIXED_M * (OPUS_BP_FIXED_K / 128);
    if constexpr (requires { a.stride_sfb_batch; })
        a.stride_sfb_batch = (OPUS_BP_FIXED_N / 128) * (OPUS_BP_FIXED_K / 128);
}
#endif

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_blockscale_mxscale_bpreshuffle_panel_half_panel {
template <class T, class Gmem>
__device__ inline auto load_sfa_vector(Gmem &g_sfa, int gmem_offset, int valid_rows) {
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
}

namespace opus_gemm_blockscale_mxscale_bpreshuffle_register_prefetch_mfma16_output {
template <class T, bool ContiguousWaveN = false, class Accumulators, class Gmem>
__device__ __forceinline__ void store_mfma16_packed(const Accumulators &c, Gmem gc, int lane,
                                                    int wm, int wn, int stride_c) {
    using namespace opus;
    static_assert(T::W_M == 16 && T::W_N == 16);
    const int side = (lane / 16) % 2;
    static_for<T::E_M * T::E_N / 2>([&](auto pair) {
        constexpr int ci = decltype(pair)::value * 2;
        const auto first = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const auto second = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci + 1]));
        const unsigned own0 = side ? second[0] : first[0];
        const unsigned own1 = side ? second[1] : first[1];
        const unsigned src0 = side ? first[0] : second[0];
        const unsigned src1 = side ? first[1] : second[1];
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, src0);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, src1);
        vector_t<unsigned, 4> packed;
        packed[0] = side ? other0 : own0;
        packed[1] = side ? other1 : own1;
        packed[2] = side ? own0 : other0;
        packed[3] = side ? own1 : other1;
        const int m = (ci + side) / T::E_N, n = (ci + side) % T::E_N;
        const int rr = (m * T::T_M + wm) * 16 + lane % 16;
        const int nn = (ContiguousWaveN ? n + wn * T::E_N : n * T::T_N + wn) * 16 + (lane / 32) * 8;
        store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed), rr * stride_c + nn, 0,
                 number<T::STORE_CACHE>{});
    });
    if constexpr (T::E_M * T::E_N % 2 != 0) {
        constexpr int ci = T::E_M * T::E_N - 1;
        const auto words = __builtin_bit_cast(vector_t<unsigned, 2>, cast<bf16_t>(c[ci]));
        const unsigned other0 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[0]);
        const unsigned other1 = __builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, words[1]);
        if (side == 0) {
            vector_t<unsigned, 4> packed{words[0], words[1], other0, other1};
            const int rr = ((ci / T::E_N) * T::T_M + wm) * 16 + lane % 16;
            const int nn = (ContiguousWaveN ? (ci % T::E_N) + wn * T::E_N : (ci % T::E_N) * T::T_N + wn) * 16 + (lane / 32) * 8;
            store<8>(gc, __builtin_bit_cast(vector_t<bf16_t, 8>, packed), rr * stride_c + nn, 0,
                     number<T::STORE_CACHE>{});
        }
    }
}
} // namespace opus_gemm_blockscale_mxscale_bpreshuffle_register_prefetch_mfma16_output

template <class T>
__device__ inline auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_ga_scale(int lane_id, int wave_id_m,
                                                                    int wave_id_n, int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
    constexpr auto ga_block_shape =
        opus::make_tuple(opus::number<T::HALF_B_M / threads_m_per_block>{}, opus::number<T::T_N>{},
                         opus::number<threads_m_per_wave>{}, opus::number<T::T_M>{},
                         opus::number<threads_k>{}, opus::number<T::VEC_A>{});
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,
                                                       lane_id % threads_k}));
}

template <class T>
__device__ inline constexpr auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_gb_scale(int lane_id, int wave_id_m,
                                                                    int wave_id_n, int stride_b) {
    constexpr int lanes_n = T::W_N;
    constexpr int lanes_k = T::WARP_SIZE / lanes_n;
    constexpr int k_per_wave_load = lanes_k * T::VEC_B;
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
    constexpr int k_repeats = T::B_K / k_per_wave_load;
    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::T_N>{}, opus::number<T::T_M>{}, opus::number<n_repeats>{},
        opus::number<k_repeats>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
    constexpr auto gb_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template <class T>
__device__ inline constexpr auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_ra_scale(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape =
        opus::make_tuple(opus::number<T::E_M>{}, opus::number<T::T_M>{}, opus::number<T::T_M>{},
                         opus::number<T::W_M / T::T_M>{}, opus::number<T::E_K>{},
                         opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
                         opus::number<T::WARP_SIZE / T::W_M>{}, opus::number<T::VEC_A>{});
    constexpr auto ra_block_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{},
                                          opus::p_dim{}, opus::y_dim{}));
    const int lane_id_m = lane_id % T::W_M;
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(
            ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M,
                                                       lane_id_m / T::T_M, lane_id / T::W_M}));
}

template <class T>
__device__ inline constexpr auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_rb_scale(int lane_id, int wave_id_n) {
    constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{}, opus::number<T::T_N>{}, opus::number<T::E_K>{},
        opus::number<k_vectors>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(
            rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

template <class T>
__device__ inline auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_sa_scale(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
    constexpr auto sa_block_shape =
        opus::make_tuple(opus::number<T::smem_m_rep / num_waves>{}, opus::number<T::T_N>{},
                         opus::number<T::T_M>{}, opus::number<T::VEC_A>{});
    constexpr auto sa_block_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(
            sa_block_dim, sa_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template <class T>
__device__ inline constexpr auto
opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_sb_scale(int wave_id_m, int wave_id_n) {
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
    constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
    constexpr int k_repeats = T::B_K / k_per_wave_load;
    constexpr auto sb_block_shape =
        opus::make_tuple(opus::number<T::T_N>{}, opus::number<T::T_M>{}, opus::number<n_repeats>{},
                         opus::number<k_repeats>{}, opus::number<T::VEC_B>{});
    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(
            sb_block_dim, sb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}
#endif

template <auto OpusValue0>
constexpr int opus_gemm_blockscale_mxscale_bpreshuffle_4wave_pin_agpr_A0_AGPR_BASE = OpusValue0;

namespace opus_gemm_bpreshuffle_ring_pin_detail {

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template <int Vec, int CHUNK_COUNT, int AGPR_BASE, class Chunk, class Mem, class Layout>
__device__ inline void load_operand_chunks_pinned(Chunk *chunks, Mem &smem, const Layout &layout) {
    constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
    static_assert(AGPR_BASE >= 0);
    static_assert(AGPR_BASE + CHUNK_COUNT * dwords_per_chunk <= 128);
    const auto offsets = opus::layout_to_offsets<Vec>(layout);
    opus::static_for<CHUNK_COUNT>([&](auto i) {
        constexpr int chunk = decltype(i)::value;
        [[clang::amdgpu_pin_agpr(AGPR_BASE + chunk * dwords_per_chunk)]] chunks[chunk] =
            opus::load<Vec>(smem, offsets[chunk]);
    });
}

template <int Vec, int OFFSET_BEGIN, class Reg, class Fragment, class Mem, class Offsets>
__device__ inline void load_operand_fragment_pinned(Reg reg, Fragment &fragment, Mem &smem,
                                                    const Offsets &offsets) {
    using Chunk = decltype(opus::load<Vec>(smem, offsets[OFFSET_BEGIN]));
    constexpr int dwords_per_chunk = sizeof(Chunk) / sizeof(opus::u32_t);
    constexpr int physical_reg = decltype(reg)::value;
    static_assert(physical_reg >= 0 && physical_reg < 128);
    opus::array<Chunk, 2> values;
    opus::static_for<2>([&](auto i) {
        constexpr int chunk = decltype(i)::value;
        [[clang::amdgpu_pin_agpr(physical_reg + chunk * dwords_per_chunk)]] values[chunk] =
            opus::load<Vec>(smem, offsets[OFFSET_BEGIN + chunk]);
    });
    Fragment result;
    opus::set_slice(result, values[0], opus::number<0>{}, opus::number<16>{});
    opus::set_slice(result, values[1], opus::number<16>{}, opus::number<32>{});
    [[clang::amdgpu_pin_agpr(physical_reg)]] fragment = result;
}

template <int Vec, int OFFSET_BEGIN, class Fragment, class Mem, class Offsets>
__device__ inline Fragment load_operand_fragment_staged(Mem &smem, const Offsets &offsets) {
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

template <class T>
__device__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n,
                                            int stride_a) {
    constexpr int threads_k = T::B_K / T::VEC_A;
    constexpr int threads_m_per_block = T::BLOCK_SIZE / threads_k;
    constexpr int threads_m_per_wave = T::WARP_SIZE / threads_k;
    constexpr auto ga_block_shape =
        opus::make_tuple(opus::number<T::HALF_B_M / threads_m_per_block>{}, opus::number<T::T_N>{},
                         opus::number<threads_m_per_wave>{}, opus::number<T::T_M>{},
                         opus::number<threads_k>{}, opus::number<T::VEC_A>{});
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, lane_id / threads_k, wave_id_m,
                                                       lane_id % threads_k}));
}

template <class T>
__device__ inline constexpr auto make_layout_gb_scale(int lane_id, int wave_id_m, int wave_id_n,
                                                      int stride_b) {
    constexpr int lanes_n = T::W_N;
    constexpr int lanes_k = T::WARP_SIZE / lanes_n;
    constexpr int k_per_wave_load = lanes_k * T::VEC_B;
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * lanes_n);
    constexpr int k_repeats = T::B_K / k_per_wave_load;
    constexpr auto gb_block_shape = opus::make_tuple(
        opus::number<T::T_N>{}, opus::number<T::T_M>{}, opus::number<n_repeats>{},
        opus::number<k_repeats>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
    constexpr auto gb_block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
                         opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        gb_block_shape,
        opus::unfold_x_stride(gb_block_dim, gb_block_shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(gb_block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template <class T>
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n,
                                                        int stride_sfa) {
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{}, opus::number<T::T_N>{},
                         opus::number<T::T_M>{}, opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
                         opus::number<T::B_M / T::VEC_SCALE_A>{}, opus::number<T::VEC_SCALE_A>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const int lane_id_m = (lane_id & 8) | ((lane_id & 3) << 1) | ((lane_id & 4) >> 2);
    return opus::make_layout<T::VEC_SCALE_A>(
        block_shape, opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(block_dim,
                             opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}

template <class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id) {
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::SCALE_N_HALVES>{},
                         opus::number<T::SCALE_PANEL_K_CAPACITY>{}, opus::number<1>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}),
                                                opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<1>(
        block_shape, opus::unfold_x_stride(block_dim, block_shape, opus::tuple{0_I, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}

template <class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape =
        opus::make_tuple(opus::number<T::E_M>{}, opus::number<T::T_M>{}, opus::number<T::T_M>{},
                         opus::number<T::W_M / T::T_M>{}, opus::number<T::E_K>{},
                         opus::number<T::W_M * T::W_K / T::WARP_SIZE / T::VEC_A>{},
                         opus::number<T::WARP_SIZE / T::W_M>{}, opus::number<T::VEC_A>{});
    constexpr auto ra_block_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{},
                                          opus::p_dim{}, opus::y_dim{}));
    const int lane_id_m = lane_id % T::W_M;
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(
            ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M,
                                                       lane_id_m / T::T_M, lane_id / T::W_M}));
}

template <class T>
__device__ inline constexpr auto make_layout_rb_scale(int lane_id, int wave_id_n) {
    constexpr int k_vectors = T::W_N * T::W_K / (T::WARP_SIZE * T::VEC_B);
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{}, opus::number<T::T_N>{}, opus::number<T::E_K>{},
        opus::number<k_vectors>{}, opus::number<T::WARP_SIZE>{}, opus::number<T::VEC_B>{});
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(
            rb_block_dim, rb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}

template <class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
    static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::T_M>{}, opus::number<T::W_M>{}, opus::number<Vec>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::p_dim{}, opus::p_dim{}),
                                                opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<Vec>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape,
                              opus::tuple{opus::number<T::VEC_SCALE_SF_PAIR>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

template <class T, int Vec = T::VEC_SCALE_SF>
__device__ inline constexpr auto make_layout_rsfb_scale() {
    static_assert(Vec == T::VEC_SCALE_SF || Vec == T::VEC_SCALE_SF_PAIR);
    constexpr auto block_shape = opus::make_tuple(opus::number<Vec>{});
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<Vec>(block_shape,
                                  opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
                                  opus::unfold_p_coord(block_dim, opus::tuple{}));
}

template <class T> __device__ inline auto make_layout_sa_scale(int wave_id_m, int wave_id_n) {
    constexpr int num_waves = T::BLOCK_SIZE / T::WARP_SIZE;
    constexpr auto sa_block_shape =
        opus::make_tuple(opus::number<T::smem_m_rep / num_waves>{}, opus::number<T::T_N>{},
                         opus::number<T::T_M>{}, opus::number<T::VEC_A>{});
    constexpr auto sa_block_dim =
        opus::make_tuple(opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        sa_block_shape,
        opus::unfold_x_stride(
            sa_block_dim, sa_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sa_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template <class T>
__device__ inline constexpr auto make_layout_sb_scale(int wave_id_m, int wave_id_n) {
    constexpr int n_repeats = T::HALF_B_N / (T::NUM_WAVES * T::W_N);
    constexpr int k_per_wave_load = (T::WARP_SIZE / T::W_N) * T::VEC_B;
    constexpr int k_repeats = T::B_K / k_per_wave_load;
    constexpr auto sb_block_shape =
        opus::make_tuple(opus::number<T::T_N>{}, opus::number<T::T_M>{}, opus::number<n_repeats>{},
                         opus::number<k_repeats>{}, opus::number<T::VEC_B>{});
    constexpr auto sb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));
    return opus::make_layout(
        sb_block_shape,
        opus::unfold_x_stride(
            sb_block_dim, sb_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(sb_block_dim, opus::tuple{wave_id_n, wave_id_m}));
}

template <class T>
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::SFA_PASSES_PER_CACHE_PANEL>{}, opus::number<T::T_N>{},
                         opus::number<T::T_M>{}, opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
                         opus::number<T::B_M / T::VEC_SF>{}, opus::number<T::VEC_SF>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    const int lane_id_m = ((lane_id & 4) << 3) | ((lane_id & 3) << 1) | ((lane_id & 8) >> 3);
    return opus::make_layout<T::VEC_SF>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape,
                              opus::tuple{opus::number<T::SFA_PANEL_PITCH>{}, 1_I}),
        opus::unfold_p_coord(block_dim,
                             opus::tuple{wave_id_n, wave_id_m, lane_id / T::W_M, lane_id_m}));
}

template <class T>
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id) {
    constexpr auto block_shape =
        opus::make_tuple(opus::number<T::SCALE_N_HALVES>{},
                         opus::number<T::SCALE_PANEL_K_CAPACITY>{}, opus::number<T::VEC_SF>{});
    constexpr auto block_dim =
        opus::make_tuple(opus::make_tuple(opus::p_dim{}), opus::make_tuple(opus::p_dim{}),
                         opus::make_tuple(opus::y_dim{}));
    return opus::make_layout<T::VEC_SF>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape,
                              opus::tuple{opus::number<T::VEC_SF>{},
                                          opus::number<T::SCALE_N_HALVES * T::VEC_SF>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id, lane_id}));
}

#endif

template <class T> struct accumulator_pin_traits {
    static constexpr bool enabled = false;
    static constexpr int agpr_base = 0;
};

template <class T, int N, int AGPR_BASE> struct pinned_accumulator_array : opus::array<T, N> {};

template <class T, int N, int AGPR_BASE>
struct accumulator_pin_traits<pinned_accumulator_array<T, N, AGPR_BASE>> {
    static constexpr bool enabled = true;
    static constexpr int agpr_base = AGPR_BASE;
};

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template <class T, int C_BEGIN, int COUNT, int EXPLICIT_AGPR_BASE = -1, int EXPLICIT_INDEX_BASE = 0,
          class MMA, class VA, class VB, class VC>
__device__ inline void mma_scale_group(MMA &mma, const VA &v_a, const VB &v_b, VC &v_c,
                                       unsigned int v_sfa, unsigned int v_sfb) {
    using tiled_mma = opus::remove_cvref_t<MMA>;
    using base_mma = typename tiled_mma::MMA;
    opus::static_for<COUNT>([&](auto i) {
        constexpr int c_index = C_BEGIN + decltype(i)::value;
        constexpr int m_repeat = c_index / T::E_N;
        constexpr int n_repeat = c_index % T::E_N;
        using pin_traits =
            opus_gemm_bpreshuffle_ring_pin_detail::accumulator_pin_traits<
                opus::remove_cvref_t<VC>>;
        constexpr int c_dwords = sizeof(typename base_mma::vtype_c) / sizeof(opus::u32_t);
        if constexpr (EXPLICIT_AGPR_BASE >= 0) {
            [[clang::amdgpu_pin_agpr(EXPLICIT_AGPR_BASE +
                                     (c_index - EXPLICIT_INDEX_BASE) * c_dwords)]] v_c[c_index] =
                base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa),
                           static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
        } else if constexpr (pin_traits::enabled) {
            [[clang::amdgpu_pin_agpr(pin_traits::agpr_base + c_index * c_dwords)]] v_c[c_index] =
                base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa),
                           static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
        } else {
            v_c[c_index] =
                base_mma{}(v_a[m_repeat], v_b[n_repeat], v_c[c_index], static_cast<int>(v_sfa),
                           static_cast<int>(v_sfb), opus::number<m_repeat>{}, opus::number<0>{});
        }
    });
}

#endif

} // namespace opus_gemm_bpreshuffle_ring_pin_detail
