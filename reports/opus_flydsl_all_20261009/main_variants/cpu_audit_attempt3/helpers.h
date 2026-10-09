#pragma once
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
using opus::operator""_I;
namespace checked {
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

}
