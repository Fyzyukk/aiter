#pragma once
#include "opus_gemm_mxscale_bpreshuffle_layout_gfx950.cuh"

namespace opus_gemm_8wave_192x256_layout {

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
        opus::number<T::T_N>{}, // 2 wave_n
        opus::number<T::T_M>{}, // 4 wave_m
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{}, // K128 groups per wave
        opus::number<T::SFA_PASSES>{}, // passes along M
        opus::number<T::SFA_THREADS_PER_GROUP>{}, // threads per K128 group
        opus::number<T::VEC_SCALE_A>{}); // 16 contiguous M scale bytes per thread

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
        opus::number<T::T_N>{}, // 2 wave_n
        opus::number<T::T_M>{}, // 4 wave_m
        opus::number<T::SFA_K_COLUMNS_PER_WAVE>{}, // K128 groups per wave
        opus::number<T::SFA_PASSES>{}, // passes along M
        opus::number<T::SFA_THREADS_PER_GROUP>{}, // threads per K128 group
        opus::number<T::VEC_SCALE_A>{}); // 16 contiguous M scale bytes per thread

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

    // Offsets count packed uint words, one per K128 group.
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

} // namespace opus_gemm_8wave_192x256_layout

namespace opus_gemm_4wave_128x128_layout {

// Load one repeat so matrix LDS reads can stay interleaved with MFMA.
template<int Vec, int Elem, class V, class Smem, class Layout, int Repeat>
__device__ inline void load_matrix_fragment(V& dst, Smem& src, const Layout& layout,
                                           opus::number<Repeat> repeat, int offset) {
    const auto fragment_layout = opus::make_layout<Vec>(
        layout.shape(), layout.stride(),
        opus::concat_tuple(opus::make_tuple(repeat),
                           opus::slice(layout.coord(), 1_I, opus::number<Layout::rank>{})));
    opus::set_slice(dst, opus::load<Vec>(src, fragment_layout + offset),
                    opus::number<Repeat * Elem>{}, opus::number<(Repeat + 1) * Elem>{});
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

} // namespace opus_gemm_4wave_128x128_layout

namespace opus_gemm_4wave_64x128_layout {

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

} // namespace opus_gemm_4wave_64x128_layout
