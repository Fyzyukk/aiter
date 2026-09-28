#include <hip/hip_runtime.h>
#include <opus/opus.hpp>
#include <iostream>
#include <cstdlib>
using opus::operator""_I;
#include "/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_9021_9030_style_20260928/9021/after/traits_runtime.cuh"
#include "/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_9021_9030_style_20260928/9022/after/traits_runtime.cuh"
#include "/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_9021_9030_style_20260928/9023/after/traits_runtime.cuh"
#include "/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_9021_9030_style_20260928/9024/after/traits_runtime.cuh"
#include "/root/workspace/aiter-opus-mxfp8-bpreshuffle/reports/opus_9021_9030_style_20260928/9030/after/traits_runtime.cuh"
namespace reference {
template<class T>
__host__ inline auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
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
}
namespace opus_gemm_4wave_128x128_layout {

template<class T>
__host__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
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

    const int matrix_row = wave_id_m * T::W_M + lane_id % T::W_M;
    const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
    const int producer_m = matrix_row % T::T_M;
    const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}

template<class T, int Pass>
__host__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    static_assert(Pass >= 0 && Pass < T::SFA_PASSES);
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_PANEL>{},
        opus::number<T::SFA_VECTORS_PER_GROUP>{},
        opus::number<T::VEC_SCALE_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    // Preserve the original pass and wave starts when a group crosses a wave boundary.
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
__host__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    static_assert(Pass >= 0 && Pass < T::SFA_PASSES);
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_PANEL>{},
        opus::number<T::SFA_VECTORS_PER_GROUP>{},
        opus::number<T::VEC_SCALE_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    // Preserve the original pass and wave starts when a group crosses a wave boundary.
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
__host__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
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
__host__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfb) {
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
__host__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id_m, int wave_id_n) {
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
__host__ inline constexpr auto make_layout_rsfb_scale() {
    constexpr auto block_shape = opus::make_tuple(opus::number<T::B_SCALE_PACKS>{}, 1_I);
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SCALE_PANEL>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{}));
}

} // namespace opus_gemm_4wave_128x128_layout

namespace opus_gemm_4wave_160x128_layout {

template<class T>
__host__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
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

    const int matrix_row = wave_id_m * T::W_M + lane_id % T::W_M;
    const int producer_n = matrix_row / (T::A_ROWS_PER_WAVE * T::T_M);
    const int producer_m = matrix_row % T::T_M;
    const int producer_lane_row = (matrix_row % (T::A_ROWS_PER_WAVE * T::T_M)) / T::T_M;

    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape, opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{producer_n, producer_m, producer_lane_row, lane_id / T::W_M}));
}

template<class T, int Pass>
__host__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    static_assert(Pass >= 0 && Pass < T::SFA_PASSES);
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_PANEL>{},
        opus::number<T::SFA_VECTORS_PER_GROUP>{},
        opus::number<T::VEC_SCALE_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    // Preserve the original pass and wave starts when a group crosses a wave boundary.
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
__host__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    static_assert(Pass >= 0 && Pass < T::SFA_PASSES);
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_PANEL>{},
        opus::number<T::SFA_VECTORS_PER_GROUP>{},
        opus::number<T::VEC_SCALE_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    // Preserve the original pass and wave starts when a group crosses a wave boundary.
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
__host__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
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
__host__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfb) {
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
__host__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id_m, int wave_id_n) {
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
__host__ inline constexpr auto make_layout_rsfb_scale() {
    constexpr auto block_shape = opus::make_tuple(opus::number<T::B_SCALE_PACKS>{}, 1_I);
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SCALE_PANEL>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{}));
}

} // namespace opus_gemm_4wave_160x128_layout

namespace opus_gemm_4wave_64x128_layout {

using opus::operator""_I;

template<class T>
__host__ inline constexpr auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::A_LOAD_PASSES>{},
        opus::number<T::T_N>{},
        opus::number<T::A_ROWS_PER_WAVE>{},
        opus::number<T::T_M>{},
        opus::number<T::A_K_VECTORS>{},
        opus::number<T::VEC_A>{});
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int producer_row = lane_id / T::A_K_VECTORS;
    const int producer_xor = (producer_row & 1) | ((producer_row & 2) << 1);
    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, producer_row, wave_id_m, (lane_id % T::A_K_VECTORS) ^ producer_xor}));
}

template<class T>
__host__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::T_M>{},
        opus::number<T::A_ROWS_PER_WAVE>{},
        opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
        opus::number<T::E_K>{},
        opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
        opus::number<T::A_LANES_K>{},
        opus::number<T::VEC_A>{});
    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;
    const int consumer_row = lane_id_m / T::T_M;
    const int chunk_xor = (consumer_row & 2) >> 1;
    const int lane_k = (lane_id / T::W_M) ^ (consumer_row & 1);
    // XOR selects the first K64 chunk and reverses the free chunk stride when needed.
    const int chunk_stride = (1 - 2 * chunk_xor) * T::A_CHUNK_STRIDE;
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, opus::number<T::B_K>{}, opus::number<T::A_CHUNK_STRIDE>{}, chunk_stride, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, consumer_row, chunk_xor, lane_k}));
}

template<class T>
__host__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
        opus::number<T::E_M>{},
        opus::number<T::SFA_PRODUCERS_PER_GROUP>{},
        opus::number<T::VEC_SCALE_A>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SCALE_A>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_PRODUCERS_PER_GROUP, lane_id % T::SFA_PRODUCERS_PER_GROUP}));
}

template<class T>
__host__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
        opus::number<T::SFA_PRODUCERS_PER_GROUP>{},
        opus::number<T::SFA_PACK_CHUNKS>{},
        opus::number<T::VEC_SCALE_PACK_A>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}));

    // Offsets count u16 values containing both M-repeat scale bytes.
    return opus::make_layout<T::VEC_SCALE_PACK_A>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_ROWS_PER_REPEAT>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_PRODUCERS_PER_GROUP, lane_id % T::SFA_PRODUCERS_PER_GROUP}));
}

template<class T>
__host__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
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

template<class T>
__host__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::WARP_SIZE>{},
        1_I);
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template<class T>
__host__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::WARP_SIZE>{},
        1_I);
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    // Offsets count packed u32 words, one per K128 group.
    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

__host__ inline constexpr auto make_layout_rsfb_scale() {
    constexpr auto block_shape = opus::make_tuple(1_I);
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{}));
}

} // namespace opus_gemm_4wave_64x128_layout

namespace opus_gemm_4wave_64x64_layout {

using opus::operator""_I;

template<class T>
__host__ inline constexpr auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
    constexpr auto ga_block_shape = opus::make_tuple(
        opus::number<T::A_LOAD_PASSES>{},
        opus::number<T::T_N>{},
        opus::number<T::A_ROWS_PER_WAVE>{},
        opus::number<T::T_M>{},
        opus::number<T::A_K_VECTORS>{},
        opus::number<T::VEC_A>{});
    constexpr auto ga_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int producer_row = lane_id / T::A_K_VECTORS;
    const int producer_xor = (producer_row & 1) | ((producer_row & 2) << 1);
    return opus::make_layout<T::VEC_A>(
        ga_block_shape,
        opus::unfold_x_stride(ga_block_dim, ga_block_shape, opus::tuple{stride_a, 1_I}),
        opus::unfold_p_coord(ga_block_dim, opus::tuple{wave_id_n, producer_row, wave_id_m, (lane_id % T::A_K_VECTORS) ^ producer_xor}));
}

template<class T>
__host__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
    constexpr auto ra_block_shape = opus::make_tuple(
        opus::number<T::E_M>{},
        opus::number<T::T_M>{},
        opus::number<T::T_M>{},
        opus::number<T::A_ROWS_PER_WAVE>{},
        opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
        opus::number<T::E_K>{},
        opus::number<T::A_CHUNKS_PER_FRAGMENT>{},
        opus::number<T::A_LANES_K>{},
        opus::number<T::VEC_A>{});
    constexpr auto ra_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    const int lane_id_m = lane_id % T::W_M;
    const int consumer_row = lane_id_m / T::T_M;
    const int chunk_xor = (consumer_row & 2) >> 1;
    const int lane_k = (lane_id / T::W_M) ^ (consumer_row & 1);
    // XOR selects the first K64 chunk and reverses the free chunk stride when needed.
    const int chunk_stride = (1 - 2 * chunk_xor) * T::A_CHUNK_STRIDE;
    return opus::make_layout<T::VEC_A>(
        ra_block_shape,
        opus::unfold_x_stride(ra_block_dim, ra_block_shape,
            opus::tuple{opus::number<T::smem_linear_wave + T::smem_padding>{}, opus::number<T::B_K>{}, opus::number<T::A_CHUNK_STRIDE>{}, chunk_stride, 1_I}),
        opus::unfold_p_coord(ra_block_dim, opus::tuple{wave_id_m, lane_id_m % T::T_M, consumer_row, chunk_xor, lane_k}));
}

template<class T>
__host__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
        opus::number<T::E_M>{},
        opus::number<T::SFA_PRODUCERS_PER_GROUP>{},
        opus::number<T::VEC_SCALE_A>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<T::VEC_SCALE_A>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_PRODUCERS_PER_GROUP, lane_id % T::SFA_PRODUCERS_PER_GROUP}));
}

template<class T>
__host__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{},
        opus::number<T::SFA_PRODUCERS_PER_GROUP>{},
        opus::number<T::SFA_PACK_CHUNKS>{},
        opus::number<T::VEC_SCALE_PACK_A>{});
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}, opus::y_dim{}));

    // Offsets count u16 values containing both M-repeat scale bytes.
    return opus::make_layout<T::VEC_SCALE_PACK_A>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SFA_ROWS_PER_REPEAT>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id / T::SFA_PRODUCERS_PER_GROUP, lane_id % T::SFA_PRODUCERS_PER_GROUP}));
}

template<class T>
__host__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
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

template<class T>
__host__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::WARP_SIZE>{},
        1_I);
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

template<class T>
__host__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::T_N>{},
        opus::number<T::T_M>{},
        opus::number<T::WARP_SIZE>{},
        1_I);
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    // Offsets count packed u32 words, one per K128 group.
    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_n, wave_id_m, lane_id}));
}

__host__ inline constexpr auto make_layout_rsfb_scale() {
    constexpr auto block_shape = opus::make_tuple(1_I);
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{}));
}

} // namespace opus_gemm_4wave_64x64_layout

namespace opus_gemm_8wave_192x256_large_output_layout {

template<class T>
__host__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
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

template<class T, int Pass>
__host__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
    static_assert(Pass >= 0 && Pass < T::SFA_PASSES);
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_PANEL>{},
        opus::number<T::SFA_ROW_VECTORS>{},
        opus::number<T::VEC_SCALE_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    // Preserve the original pass and wave starts when a group crosses a wave boundary.
    const int wave_vector_begin = (Pass * T::NUM_WAVES + wave_id_n * T::T_M + wave_id_m) * T::WARP_SIZE;
    const int lane_vector = wave_vector_begin % T::SFA_ROW_VECTORS + lane_id;
    const int local_k_group = wave_vector_begin / T::SFA_ROW_VECTORS + lane_vector / T::SFA_ROW_VECTORS;
    const int row_vector = lane_vector % T::SFA_ROW_VECTORS;

    return opus::make_layout<T::VEC_SCALE_A>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfa, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{local_k_group, row_vector}));
}

template<class T, int Pass>
__host__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
    static_assert(Pass >= 0 && Pass < T::SFA_PASSES);
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_PANEL>{},
        opus::number<T::SFA_ROW_VECTORS>{},
        opus::number<T::VEC_SCALE_A>{});

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));

    // Preserve the original pass and wave starts when a group crosses a wave boundary.
    const int wave_vector_begin = (Pass * T::NUM_WAVES + wave_id_n * T::T_M + wave_id_m) * T::WARP_SIZE;
    const int lane_vector = wave_vector_begin % T::SFA_ROW_VECTORS + lane_id;
    const int local_k_group = wave_vector_begin / T::SFA_ROW_VECTORS + lane_vector / T::SFA_ROW_VECTORS;
    const int row_vector = lane_vector % T::SFA_ROW_VECTORS;

    return opus::make_layout<T::VEC_SCALE_A>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::B_M>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{local_k_group, row_vector}));
}

template<class T>
__host__ inline constexpr auto make_layout_rsfa_scale(int lane_id, int wave_id_m) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::E_M>{}, // 3 M repeats
        opus::number<T::T_M>{}, // 4 wave_m
        opus::number<T::W_M>{}, // 16 M lanes
        1_I); // one scale byte per M repeat

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{wave_id_m, lane_id % T::W_M}));
}

template<class T>
__host__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfb) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_N_HALVES>{},
        opus::number<T::SFB_WAVES_PER_HALF>{},
        opus::number<T::WARP_SIZE>{},
        1_I);

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int half_n = wave_id_n * (T::T_M / T::SFB_WAVES_PER_HALF) + wave_id_m / T::SFB_WAVES_PER_HALF;
    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{stride_sfb, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{half_n, wave_id_m % T::SFB_WAVES_PER_HALF, lane_id}));
}

template<class T>
__host__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id_m, int wave_id_n) {
    constexpr auto block_shape = opus::make_tuple(
        opus::number<T::SCALE_N_HALVES>{},
        opus::number<T::SFB_WAVES_PER_HALF>{},
        opus::number<T::WARP_SIZE>{},
        1_I);

    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}));

    const int half_n = wave_id_n * (T::T_M / T::SFB_WAVES_PER_HALF) + wave_id_m / T::SFB_WAVES_PER_HALF;
    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SCALE_PANEL>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{half_n, wave_id_m % T::SFB_WAVES_PER_HALF, lane_id}));
}

template<class T>
__host__ inline constexpr auto make_layout_rsfb_scale() {
    constexpr auto block_shape = opus::make_tuple(opus::number<T::B_SCALE_PACKS>{}, 1_I);
    constexpr auto block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{opus::number<T::SCALE_PANEL>{}, 1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{}));
}

} // namespace opus_gemm_8wave_192x256_large_output_layout


long long checks = 0;
void eq(long long got, long long expected, int kid, const char* kind, int tid, int index) {
    ++checks;
    if (got != expected) { std::cerr << kid << " " << kind << " thread=" << tid << " index=" << index << " got=" << got << " expected=" << expected << "\n"; std::exit(2); }
}
template<class T> int old_ra(int wm, int lane, int repeat, int chunk) {
    const int matrix_row = repeat*T::T_M*T::W_M + wm*T::W_M + lane%T::W_M;
    const int rows_per_pass = T::BLOCK_SIZE/(T::B_K/T::VEC_A);
    const int pass = matrix_row/rows_per_pass, remainder = matrix_row%rows_per_pass;
    const int producer_n = remainder/(8*T::T_M), producer_m = remainder%T::T_M;
    const int producer_lane_row = (remainder%(8*T::T_M))/T::T_M;
    return (pass*T::NUM_WAVES+producer_n*T::T_M+producer_m)*(T::smem_linear_wave+T::smem_padding)+producer_lane_row*T::B_K+(lane/T::W_M)*T::VEC_A+chunk*64;
}
template<class T> int old_xor_ra(int wm, int lane, int repeat, int chunk) {
    const int lane_m=lane%T::W_M, consumer_row=lane_m/T::T_M;
    const int x=(consumer_row&1)|((consumer_row&2)<<1);
    const int segment=(repeat*T::T_M+wm)*T::T_M+lane_m%T::T_M;
    const int k=(chunk*(T::WARP_SIZE/T::W_M)+lane/T::W_M)^x;
    return segment*(T::smem_linear_wave+T::smem_padding)+consumer_row*T::B_K+k*T::VEC_A;
}
void check_9021() {
    using namespace opus;
    using T = opus_gemm_mxscale_bpreshuffle_4wave_128x128_traits_gfx950;
    namespace L = opus_gemm_4wave_128x128_layout;
    constexpr int kid=9021, stride_a=8192, stride_sfa=1472, stride_sfb=128, tile_row=384, tile_col=256;
    const auto start=checks;
    auto mma = make_tiled_mma<fp8_t,fp8_t,fp32_t>(seq<T::E_M,T::E_N,T::E_K>{},seq<T::T_M,T::T_N,T::T_K>{},seq<T::W_M,T::W_N,T::W_K>{},mfma_adaptor_swap_ab{});
    for(int tid=0;tid<T::BLOCK_SIZE;++tid) {
        const int lane=tid%T::WARP_SIZE, wave=tid/T::WARP_SIZE, wm=wave%T::T_M, wn=wave/T::T_M;
        const auto ra=L::make_layout_ra_scale<T>(lane,wm);
        for(int stage=0;stage<T::NUM_STAGES;++stage) {
            const auto offsets=layout_to_offsets<T::VEC_A>(ra+stage*T::A_STAGE);
            for(int m=0;m<T::E_M;++m) for(int c=0;c<T::A_CHUNKS_PER_FRAGMENT;++c)
                eq(offsets[m*T::A_CHUNKS_PER_FRAGMENT+c],old_ra<T>(wm,lane,m,c)+stage*T::A_STAGE,kid,"RA",tid,m*2+c);
        }
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass=decltype(pass_i)::value;
            const auto gsa=layout_to_offsets<T::VEC_SCALE_A>(L::make_layout_gsfa_scale<T,pass>(lane,wm,wn,stride_sfa));
            const auto ssa=layout_to_offsets<T::VEC_SCALE_A>(L::make_layout_ssfa_scale<T,pass>(lane,wm,wn));
            const int index=(tid+pass*T::BLOCK_SIZE)*16, group=index/T::B_M, local_row=index%T::B_M;
            for(int byte=0;byte<T::VEC_SCALE_A;++byte) {
                eq(ssa[0]+byte,index+byte,kid,"SSFA",tid,pass*16+byte);
                eq(gsa[0]+byte,group*stride_sfa+local_row+byte,kid,"GSFA",tid,pass*16+byte);
            }
            for(int groups : {1,31,32,33,64,65,128}) for(int begin=0;begin<groups;begin+=T::SCALE_PANEL) {
                const bool valid=index<T::SFA_BYTES && begin+group<groups;
                eq(valid,ssa[0]<T::SFA_BYTES && begin+ssa[0]/T::B_M<groups,kid,"SFA guard",tid,pass);
                if(valid) eq(tile_row+gsa[0]+begin*stride_sfa,(begin+group)*stride_sfa+tile_row+local_row,kid,"SFA tile/panel",tid,pass);
            }
        });
        const auto rsa=layout_to_offsets<1>(L::make_layout_rsfa_scale<T>(lane,wm));
        for(int m=0;m<T::E_M;++m) for(int tile=0;tile<128;++tile)
            eq(rsa[m]+(tile&(T::SCALE_PANEL-1))*T::B_M,(tile&(T::SCALE_PANEL-1))*T::B_M+wm*T::W_M+lane%T::W_M+m*T::T_M*T::W_M,kid,"RSFA",tid,m*128+tile);
        const auto gsb=layout_to_offsets<1>(L::make_layout_gsfb_scale<T>(lane,wm,wn,stride_sfb));
        const auto ssb=layout_to_offsets<1>(L::make_layout_ssfb_scale<T>(lane,wm,wn));
        const auto rsb=layout_to_offsets<1>(L::make_layout_rsfb_scale<T>());
        eq(gsb[0],tid,kid,"GSFB",tid,0); eq(ssb[0],tid,kid,"SSFB bytes",tid,0);
        for(int groups : {1,32,33,128}) for(int begin=0;begin<groups;begin+=T::SCALE_PANEL)
            if(tid<T::SCALE_PANEL && begin+tid<groups)
                eq((tile_col/T::GROUP_N)*stride_sfb+gsb[0]+begin,(tile_col/T::GROUP_N)*stride_sfb+begin+tid,kid,"SFB tile/panel",tid,begin);
        for(int half=0;half<T::B_SCALE_PACKS;++half) for(int tile=0;tile<128;++tile)
            eq(rsb[half]+(tile&(T::SCALE_PANEL-1)),half*T::SCALE_PANEL+(tile&(T::SCALE_PANEL-1)),kid,"RSFB",tid,half*128+tile);
        const int c_stride=T::C_LDS_ROW_STRIDE_ELEMS;
        const auto coord=opus::make_tuple(wm,lane%mma.grpn_c,wn,lane/mma.grpn_c);
        const auto gc=layout_to_offsets<T::VEC_C>(partition_layout_c<T::VEC_C>(mma,opus::make_tuple(c_stride,1_I),coord));
        for(int m=0;m<T::E_M;++m) for(int n=0;n<T::E_N;++n)
            eq(gc[m*T::E_N+n],(m*T::T_M*T::W_M+wm*T::W_M+lane%T::W_M)*c_stride+n*T::T_N*T::W_N+wn*T::W_N+(lane/T::W_M)*T::VEC_C,kid,"C layout",tid,m*T::E_N+n);
    }
    std::cout<<kid<<" "<<(checks-start)<<" effective address/guard checks passed\n";
}
void check_9022() {
    using namespace opus;
    using T = opus_gemm_mxscale_bpreshuffle_4wave_160x128_traits_gfx950;
    namespace L = opus_gemm_4wave_160x128_layout;
    constexpr int kid=9022, stride_a=8192, stride_sfa=1472, stride_sfb=128, tile_row=384, tile_col=256;
    const auto start=checks;
    auto mma = make_tiled_mma<fp8_t,fp8_t,fp32_t>(seq<T::E_M,T::E_N,T::E_K>{},seq<T::T_M,T::T_N,T::T_K>{},seq<T::W_M,T::W_N,T::W_K>{},mfma_adaptor_swap_ab{});
    for(int tid=0;tid<T::BLOCK_SIZE;++tid) {
        const int lane=tid%T::WARP_SIZE, wave=tid/T::WARP_SIZE, wm=wave%T::T_M, wn=wave/T::T_M;
        const auto ra=L::make_layout_ra_scale<T>(lane,wm);
        for(int stage=0;stage<T::NUM_STAGES;++stage) {
            const auto offsets=layout_to_offsets<T::VEC_A>(ra+stage*T::A_STAGE);
            for(int m=0;m<T::E_M;++m) for(int c=0;c<T::A_CHUNKS_PER_FRAGMENT;++c)
                eq(offsets[m*T::A_CHUNKS_PER_FRAGMENT+c],old_ra<T>(wm,lane,m,c)+stage*T::A_STAGE,kid,"RA",tid,m*2+c);
        }
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass=decltype(pass_i)::value;
            const auto gsa=layout_to_offsets<T::VEC_SCALE_A>(L::make_layout_gsfa_scale<T,pass>(lane,wm,wn,stride_sfa));
            const auto ssa=layout_to_offsets<T::VEC_SCALE_A>(L::make_layout_ssfa_scale<T,pass>(lane,wm,wn));
            const int index=(tid+pass*T::BLOCK_SIZE)*16, group=index/T::B_M, local_row=index%T::B_M;
            for(int byte=0;byte<T::VEC_SCALE_A;++byte) {
                eq(ssa[0]+byte,index+byte,kid,"SSFA",tid,pass*16+byte);
                eq(gsa[0]+byte,group*stride_sfa+local_row+byte,kid,"GSFA",tid,pass*16+byte);
            }
            for(int groups : {1,31,32,33,64,65,128}) for(int begin=0;begin<groups;begin+=T::SCALE_PANEL) {
                const bool valid=index<T::SFA_BYTES && begin+group<groups;
                eq(valid,ssa[0]<T::SFA_BYTES && begin+ssa[0]/T::B_M<groups,kid,"SFA guard",tid,pass);
                if(valid) eq(tile_row+gsa[0]+begin*stride_sfa,(begin+group)*stride_sfa+tile_row+local_row,kid,"SFA tile/panel",tid,pass);
            }
        });
        const auto rsa=layout_to_offsets<1>(L::make_layout_rsfa_scale<T>(lane,wm));
        for(int m=0;m<T::E_M;++m) for(int tile=0;tile<128;++tile)
            eq(rsa[m]+(tile&(T::SCALE_PANEL-1))*T::B_M,(tile&(T::SCALE_PANEL-1))*T::B_M+wm*T::W_M+lane%T::W_M+m*T::T_M*T::W_M,kid,"RSFA",tid,m*128+tile);
        const auto gsb=layout_to_offsets<1>(L::make_layout_gsfb_scale<T>(lane,wm,wn,stride_sfb));
        const auto ssb=layout_to_offsets<1>(L::make_layout_ssfb_scale<T>(lane,wm,wn));
        const auto rsb=layout_to_offsets<1>(L::make_layout_rsfb_scale<T>());
        eq(gsb[0],tid,kid,"GSFB",tid,0); eq(ssb[0],tid,kid,"SSFB bytes",tid,0);
        for(int groups : {1,32,33,128}) for(int begin=0;begin<groups;begin+=T::SCALE_PANEL)
            if(tid<T::SCALE_PANEL && begin+tid<groups)
                eq((tile_col/T::GROUP_N)*stride_sfb+gsb[0]+begin,(tile_col/T::GROUP_N)*stride_sfb+begin+tid,kid,"SFB tile/panel",tid,begin);
        for(int half=0;half<T::B_SCALE_PACKS;++half) for(int tile=0;tile<128;++tile)
            eq(rsb[half]+(tile&(T::SCALE_PANEL-1)),half*T::SCALE_PANEL+(tile&(T::SCALE_PANEL-1)),kid,"RSFB",tid,half*128+tile);
        const int c_stride=T::C_LDS_ROW_STRIDE_ELEMS;
        const auto coord=opus::make_tuple(wm,lane%mma.grpn_c,wn,lane/mma.grpn_c);
        const auto gc=layout_to_offsets<T::VEC_C>(partition_layout_c<T::VEC_C>(mma,opus::make_tuple(c_stride,1_I),coord));
        for(int m=0;m<T::E_M;++m) for(int n=0;n<T::E_N;++n)
            eq(gc[m*T::E_N+n],(m*T::T_M*T::W_M+wm*T::W_M+lane%T::W_M)*c_stride+n*T::T_N*T::W_N+wn*T::W_N+(lane/T::W_M)*T::VEC_C,kid,"C layout",tid,m*T::E_N+n);
    }
    std::cout<<kid<<" "<<(checks-start)<<" effective address/guard checks passed\n";
}
void check_9023() {
    using namespace opus;
    using T = opus_gemm_mxscale_bpreshuffle_4wave_64x128_traits_gfx950;
    namespace L = opus_gemm_4wave_64x128_layout;
    constexpr int kid=9023, stride_a=8192, stride_sfa=1472, stride_sfb=128, tile_row=384, tile_col=256;
    const auto start=checks;
    auto mma = make_tiled_mma<fp8_t,fp8_t,fp32_t>(seq<T::E_M,T::E_N,T::E_K>{},seq<T::T_M,T::T_N,T::T_K>{},seq<T::W_M,T::W_N,T::W_K>{},mfma_adaptor_swap_ab{});
    for(int tid=0;tid<T::BLOCK_SIZE;++tid) {
        const int lane=tid%T::WARP_SIZE, wave=tid/T::WARP_SIZE, wm=wave%T::T_M, wn=wave/T::T_M;
        const auto ra=L::make_layout_ra_scale<T>(lane,wm);
        for(int stage=0;stage<T::NUM_STAGES;++stage) {
            const auto offsets=layout_to_offsets<T::VEC_A>(ra+stage*T::A_STAGE);
            for(int m=0;m<T::E_M;++m) for(int c=0;c<T::A_CHUNKS_PER_FRAGMENT;++c)
                eq(offsets[m*T::A_CHUNKS_PER_FRAGMENT+c],old_xor_ra<T>(wm,lane,m,c)+stage*T::A_STAGE,kid,"RA",tid,m*2+c);
        }
        const int producer_row=lane/(T::B_K/T::VEC_A);
        const int producer_xor=(producer_row&1)|((producer_row&2)<<1);
        const auto old_ga=layout_to_offsets<T::VEC_A>(reference::make_layout_ga_scale<T>(lane^producer_xor,wm,wn,stride_a));
        const auto new_ga=layout_to_offsets<T::VEC_A>(L::make_layout_ga_scale<T>(lane,wm,wn,stride_a));
        for(int i=0;i<old_ga.size();++i) eq(new_ga[i],old_ga[i],kid,"GA XOR",tid,i);
        const auto gsa=layout_to_offsets<T::VEC_SCALE_A>(L::make_layout_gsfa_scale<T>(lane,wm,wn,stride_sfa));
        const auto ssa=layout_to_offsets<T::VEC_SCALE_PACK_A>(L::make_layout_ssfa_scale<T>(lane,wm,wn));
        const auto rsa=layout_to_offsets<1>(L::make_layout_rsfa_scale<T>(lane,wm));
        const auto gsb=layout_to_offsets<1>(L::make_layout_gsfb_scale<T>(lane,wm,wn));
        const auto ssb=layout_to_offsets<1>(L::make_layout_ssfb_scale<T>(lane,wm,wn));
        const auto rsb=layout_to_offsets<1>(L::make_layout_rsfb_scale());
        const int group=tid/T::SFA_PRODUCERS_PER_GROUP, pair=(tid%T::SFA_PRODUCERS_PER_GROUP)*T::VEC_SCALE_A;
        for(int m=0;m<T::E_M;++m) for(int byte=0;byte<T::VEC_SCALE_A;++byte)
            eq(gsa[m]+byte,group*stride_sfa+pair+m*T::SFA_ROWS_PER_REPEAT+byte,kid,"GSFA",tid,m*16+byte);
        for(int c=0;c<T::SFA_PACK_CHUNKS;++c) for(int e=0;e<T::VEC_SCALE_PACK_A;++e)
            eq(ssa[c]+e,group*T::SFA_ROWS_PER_REPEAT+pair+c*8+e,kid,"SSFA u16",tid,c*8+e);
        eq(rsa[0],wm*T::W_M+lane%T::W_M,kid,"RSFA",tid,0);
        eq(gsb[0],tid,kid,"GSFB",tid,0); eq(ssb[0],tid,kid,"SSFB u32",tid,0); eq(rsb[0],0,kid,"RSFB",tid,0);
        for(int groups : {1,32,64,65,128}) for(int begin=0;begin<groups;begin+=T::SCALE_PANEL) {
            eq(group<T::SCALE_PANEL && begin+group<groups,ssa[0]/T::SFA_ROWS_PER_REPEAT<T::SCALE_PANEL && begin+ssa[0]/T::SFA_ROWS_PER_REPEAT<groups,kid,"SFA guard",tid,begin);
            if(group<T::SCALE_PANEL && begin+group<groups)
                eq(tile_row+gsa[0]+begin*stride_sfa,(begin+group)*stride_sfa+tile_row+pair,kid,"SFA tile/panel",tid,begin);
            if(tid<T::SCALE_PANEL && begin+tid<groups)
                eq((tile_col/T::GROUP_N)*stride_sfb+gsb[0]+begin,(tile_col/T::GROUP_N)*stride_sfb+begin+tid,kid,"SFB tile/panel",tid,begin);
        }
        for(int tile=0;tile<128;++tile) {
            eq(rsa[0]+(tile&(T::SCALE_PANEL-1))*T::SFA_ROWS_PER_REPEAT,(tile&(T::SCALE_PANEL-1))*T::SFA_ROWS_PER_REPEAT+wm*T::W_M+lane%T::W_M,kid,"RSFA panel",tid,tile);
            eq(rsb[0]+(tile&(T::SCALE_PANEL-1)),tile&(T::SCALE_PANEL-1),kid,"RSFB panel",tid,tile);
        }
        const int c_stride=32768;
        const auto coord=opus::make_tuple(wm,lane%mma.grpn_c,wn,lane/mma.grpn_c);
        const auto gc=layout_to_offsets<T::VEC_C>(partition_layout_c<T::VEC_C>(mma,opus::make_tuple(c_stride,1_I),coord));
        for(int m=0;m<T::E_M;++m) for(int n=0;n<T::E_N;++n)
            eq(gc[m*T::E_N+n],(m*T::T_M*T::W_M+wm*T::W_M+lane%T::W_M)*c_stride+n*T::T_N*T::W_N+wn*T::W_N+(lane/T::W_M)*T::VEC_C,kid,"C layout",tid,m*T::E_N+n);
    }
    std::cout<<kid<<" "<<(checks-start)<<" effective address/guard checks passed\n";
}
void check_9024() {
    using namespace opus;
    using T = opus_gemm_mxscale_bpreshuffle_4wave_64x64_traits_gfx950;
    namespace L = opus_gemm_4wave_64x64_layout;
    constexpr int kid=9024, stride_a=8192, stride_sfa=1472, stride_sfb=128, tile_row=384, tile_col=256;
    const auto start=checks;
    auto mma = make_tiled_mma<fp8_t,fp8_t,fp32_t>(seq<T::E_M,T::E_N,T::E_K>{},seq<T::T_M,T::T_N,T::T_K>{},seq<T::W_M,T::W_N,T::W_K>{},mfma_adaptor_swap_ab{});
    for(int tid=0;tid<T::BLOCK_SIZE;++tid) {
        const int lane=tid%T::WARP_SIZE, wave=tid/T::WARP_SIZE, wm=wave%T::T_M, wn=wave/T::T_M;
        const auto ra=L::make_layout_ra_scale<T>(lane,wm);
        for(int stage=0;stage<T::NUM_STAGES;++stage) {
            const auto offsets=layout_to_offsets<T::VEC_A>(ra+stage*T::A_STAGE);
            for(int m=0;m<T::E_M;++m) for(int c=0;c<T::A_CHUNKS_PER_FRAGMENT;++c)
                eq(offsets[m*T::A_CHUNKS_PER_FRAGMENT+c],old_xor_ra<T>(wm,lane,m,c)+stage*T::A_STAGE,kid,"RA",tid,m*2+c);
        }
        const int producer_row=lane/(T::B_K/T::VEC_A);
        const int producer_xor=(producer_row&1)|((producer_row&2)<<1);
        const auto old_ga=layout_to_offsets<T::VEC_A>(reference::make_layout_ga_scale<T>(lane^producer_xor,wm,wn,stride_a));
        const auto new_ga=layout_to_offsets<T::VEC_A>(L::make_layout_ga_scale<T>(lane,wm,wn,stride_a));
        for(int i=0;i<old_ga.size();++i) eq(new_ga[i],old_ga[i],kid,"GA XOR",tid,i);
        const auto gsa=layout_to_offsets<T::VEC_SCALE_A>(L::make_layout_gsfa_scale<T>(lane,wm,wn,stride_sfa));
        const auto ssa=layout_to_offsets<T::VEC_SCALE_PACK_A>(L::make_layout_ssfa_scale<T>(lane,wm,wn));
        const auto rsa=layout_to_offsets<1>(L::make_layout_rsfa_scale<T>(lane,wm));
        const auto gsb=layout_to_offsets<1>(L::make_layout_gsfb_scale<T>(lane,wm,wn));
        const auto ssb=layout_to_offsets<1>(L::make_layout_ssfb_scale<T>(lane,wm,wn));
        const auto rsb=layout_to_offsets<1>(L::make_layout_rsfb_scale());
        const int group=tid/T::SFA_PRODUCERS_PER_GROUP, pair=(tid%T::SFA_PRODUCERS_PER_GROUP)*T::VEC_SCALE_A;
        for(int m=0;m<T::E_M;++m) for(int byte=0;byte<T::VEC_SCALE_A;++byte)
            eq(gsa[m]+byte,group*stride_sfa+pair+m*T::SFA_ROWS_PER_REPEAT+byte,kid,"GSFA",tid,m*16+byte);
        for(int c=0;c<T::SFA_PACK_CHUNKS;++c) for(int e=0;e<T::VEC_SCALE_PACK_A;++e)
            eq(ssa[c]+e,group*T::SFA_ROWS_PER_REPEAT+pair+c*8+e,kid,"SSFA u16",tid,c*8+e);
        eq(rsa[0],wm*T::W_M+lane%T::W_M,kid,"RSFA",tid,0);
        eq(gsb[0],tid,kid,"GSFB",tid,0); eq(ssb[0],tid,kid,"SSFB u32",tid,0); eq(rsb[0],0,kid,"RSFB",tid,0);
        for(int groups : {1,32,64,65,128}) for(int begin=0;begin<groups;begin+=T::SCALE_PANEL) {
            eq(group<T::SCALE_PANEL && begin+group<groups,ssa[0]/T::SFA_ROWS_PER_REPEAT<T::SCALE_PANEL && begin+ssa[0]/T::SFA_ROWS_PER_REPEAT<groups,kid,"SFA guard",tid,begin);
            if(group<T::SCALE_PANEL && begin+group<groups)
                eq(tile_row+gsa[0]+begin*stride_sfa,(begin+group)*stride_sfa+tile_row+pair,kid,"SFA tile/panel",tid,begin);
            if(tid<T::SCALE_PANEL && begin+tid<groups)
                eq((tile_col/T::GROUP_N)*stride_sfb+gsb[0]+begin,(tile_col/T::GROUP_N)*stride_sfb+begin+tid,kid,"SFB tile/panel",tid,begin);
        }
        for(int tile=0;tile<128;++tile) {
            eq(rsa[0]+(tile&(T::SCALE_PANEL-1))*T::SFA_ROWS_PER_REPEAT,(tile&(T::SCALE_PANEL-1))*T::SFA_ROWS_PER_REPEAT+wm*T::W_M+lane%T::W_M,kid,"RSFA panel",tid,tile);
            eq(rsb[0]+(tile&(T::SCALE_PANEL-1)),tile&(T::SCALE_PANEL-1),kid,"RSFB panel",tid,tile);
        }
        const int c_stride=32768;
        const auto coord=opus::make_tuple(wm,lane%mma.grpn_c,wn,lane/mma.grpn_c);
        const auto gc=layout_to_offsets<T::VEC_C>(partition_layout_c<T::VEC_C>(mma,opus::make_tuple(c_stride,1_I),coord));
        for(int m=0;m<T::E_M;++m) for(int n=0;n<T::E_N;++n)
            eq(gc[m*T::E_N+n],(m*T::T_M*T::W_M+wm*T::W_M+lane%T::W_M)*c_stride+n*T::T_N*T::W_N+wn*T::W_N+(lane/T::W_M)*T::VEC_C,kid,"C layout",tid,m*T::E_N+n);
    }
    std::cout<<kid<<" "<<(checks-start)<<" effective address/guard checks passed\n";
}
void check_9030() {
    using namespace opus;
    using T = opus_gemm_mxscale_bpreshuffle_8wave_192x256_large_output_traits_gfx950;
    namespace L = opus_gemm_8wave_192x256_large_output_layout;
    constexpr int kid=9030, stride_a=8192, stride_sfa=1472, stride_sfb=128, tile_row=384, tile_col=256;
    const auto start=checks;
    auto mma = make_tiled_mma<fp8_t,fp8_t,fp32_t>(seq<T::E_M,T::E_N,T::E_K>{},seq<T::T_M,T::T_N,T::T_K>{},seq<T::W_M,T::W_N,T::W_K>{},mfma_adaptor_swap_ab{});
    for(int tid=0;tid<T::BLOCK_SIZE;++tid) {
        const int lane=tid%T::WARP_SIZE, wave=tid/T::WARP_SIZE, wm=wave%T::T_M, wn=wave/T::T_M;
        const auto ra=L::make_layout_ra_scale<T>(lane,wm);
        for(int stage=0;stage<T::NUM_STAGES;++stage) {
            const auto offsets=layout_to_offsets<T::VEC_A>(ra+stage*T::A_STAGE);
            for(int m=0;m<T::E_M;++m) for(int c=0;c<T::A_CHUNKS_PER_FRAGMENT;++c)
                eq(offsets[m*T::A_CHUNKS_PER_FRAGMENT+c],old_ra<T>(wm,lane,m,c)+stage*T::A_STAGE,kid,"RA",tid,m*2+c);
        }
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass=decltype(pass_i)::value;
            const auto gsa=layout_to_offsets<T::VEC_SCALE_A>(L::make_layout_gsfa_scale<T,pass>(lane,wm,wn,stride_sfa));
            const auto ssa=layout_to_offsets<T::VEC_SCALE_A>(L::make_layout_ssfa_scale<T,pass>(lane,wm,wn));
            const int index=(tid+pass*T::BLOCK_SIZE)*16, group=index/T::B_M, local_row=index%T::B_M;
            for(int byte=0;byte<T::VEC_SCALE_A;++byte) {
                eq(ssa[0]+byte,index+byte,kid,"SSFA",tid,pass*16+byte);
                eq(gsa[0]+byte,group*stride_sfa+local_row+byte,kid,"GSFA",tid,pass*16+byte);
            }
            for(int groups : {1,31,32,33,64,65,128}) for(int begin=0;begin<groups;begin+=T::SCALE_PANEL) {
                const bool valid=index<T::SFA_BYTES && begin+group<groups;
                eq(valid,ssa[0]<T::SFA_BYTES && begin+ssa[0]/T::B_M<groups,kid,"SFA guard",tid,pass);
                if(valid) eq(tile_row+gsa[0]+begin*stride_sfa,(begin+group)*stride_sfa+tile_row+local_row,kid,"SFA tile/panel",tid,pass);
            }
        });
        const auto rsa=layout_to_offsets<1>(L::make_layout_rsfa_scale<T>(lane,wm));
        for(int m=0;m<T::E_M;++m) for(int tile=0;tile<128;++tile)
            eq(rsa[m]+(tile&(T::SCALE_PANEL-1))*T::B_M,(tile&(T::SCALE_PANEL-1))*T::B_M+wm*T::W_M+lane%T::W_M+m*T::T_M*T::W_M,kid,"RSFA",tid,m*128+tile);
        const auto gsb=layout_to_offsets<1>(L::make_layout_gsfb_scale<T>(lane,wm,wn,stride_sfb));
        const auto ssb=layout_to_offsets<1>(L::make_layout_ssfb_scale<T>(lane,wm,wn));
        const auto rsb=layout_to_offsets<1>(L::make_layout_rsfb_scale<T>());
        eq(gsb[0],(tid/T::SCALE_PANEL)*stride_sfb+tid%T::SCALE_PANEL,kid,"GSFB halves",tid,0);
        eq(ssb[0],tid,kid,"SSFB bytes",tid,0);
        if(tid<T::SFB_BYTES) for(int groups : {1,64,128}) {
            const int group=tid%T::SCALE_PANEL, half=tid/T::SCALE_PANEL;
            if(group<groups) eq((tile_col/T::GROUP_N)*stride_sfb+gsb[0],(tile_col/T::GROUP_N+half)*stride_sfb+group,kid,"SFB tile",tid,groups);
        }
        for(int half=0;half<T::B_SCALE_PACKS;++half) for(int tile=0;tile<128;++tile)
            eq(rsb[half]+(tile&(T::SCALE_PANEL-1)),half*T::SCALE_PANEL+(tile&(T::SCALE_PANEL-1)),kid,"RSFB",tid,half*128+tile);
        const int c_stride=T::C_LDS_ROW_STRIDE_ELEMS;
        const auto coord=opus::make_tuple(wm,lane%mma.grpn_c,wn,lane/mma.grpn_c);
        const auto gc=layout_to_offsets<T::VEC_C>(partition_layout_c<T::VEC_C>(mma,opus::make_tuple(c_stride,1_I),coord));
        for(int m=0;m<T::E_M;++m) for(int n=0;n<T::E_N;++n)
            eq(gc[m*T::E_N+n],(m*T::T_M*T::W_M+wm*T::W_M+lane%T::W_M)*c_stride+n*T::T_N*T::W_N+wn*T::W_N+(lane/T::W_M)*T::VEC_C,kid,"C layout",tid,m*T::E_N+n);
    }
    std::cout<<kid<<" "<<(checks-start)<<" effective address/guard checks passed\n";
}
int main(){check_9021();check_9022();check_9023();check_9024();check_9030();std::cout<<"total "<<checks<<" checks passed\n";}
