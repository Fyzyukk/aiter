// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "traits_runtime.cuh"

namespace opus_gemm_4wave_64x128_layout {

using opus::operator""_I;

template<class T>
__device__ inline constexpr auto make_layout_ra_scale(int lane_id, int wave_id_m) {
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
__device__ inline constexpr auto make_layout_gsfa_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_sfa) {
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
__device__ inline constexpr auto make_layout_ssfa_scale(int lane_id, int wave_id_m, int wave_id_n) {
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

template<class T>
__device__ inline constexpr auto make_layout_gsfb_scale(int lane_id, int wave_id_m, int wave_id_n) {
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
__device__ inline constexpr auto make_layout_ssfb_scale(int lane_id, int wave_id_m, int wave_id_n) {
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

__device__ inline constexpr auto make_layout_rsfb_scale() {
    constexpr auto block_shape = opus::make_tuple(1_I);
    constexpr auto block_dim = opus::make_tuple(opus::make_tuple(opus::y_dim{}));

    return opus::make_layout<1>(
        block_shape,
        opus::unfold_x_stride(block_dim, block_shape, opus::tuple{1_I}),
        opus::unfold_p_coord(block_dim, opus::tuple{}));
}

} // namespace opus_gemm_4wave_64x128_layout

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_scale_4wave_64x128_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void gemm_a8w8_mxfp8_scale_4wave_64x128_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;
    using ScaleWords = vector_t<D_SF_PACK, T::SFA_WORDS>;
    namespace layout_9023 = opus_gemm_4wave_64x128_layout;

    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    const unsigned grid_m = kargs.m / T::B_M;
    const unsigned grid_n = kargs.n / T::B_N;
    const unsigned total = grid_m * grid_n;
    unsigned block_m = block_id_y();
    unsigned block_n = block_id_x();
    const unsigned linear = block_m * grid_n + block_n;
    // Tile traversal depends only on grid geometry.
    if (T::B_N == T::SWIZZLE_PAIR_B_N && grid_n <= T::SWIZZLE_MAX_GRID_N &&
        (grid_n & (T::SWIZZLE_N_PARTITIONS - 1)) == 0u &&
        grid_m >= T::SWIZZLE_PAIR_MIN_GRID_M && grid_m <= T::SWIZZLE_PAIR_MAX_GRID_M) {
        const unsigned n_per_partition = grid_n / T::SWIZZLE_N_PARTITIONS;
        const unsigned full_pairs = (grid_m & ~(T::SWIZZLE_M_PAIR - 1)) * grid_n;
        if (linear < full_pairs) {
            const unsigned partition = linear & (T::SWIZZLE_PAIR_PARTITIONS - 1);
            const unsigned local = linear >> T::SWIZZLE_PAIR_SHIFT;
            block_m = (local / n_per_partition) * T::SWIZZLE_M_PAIR + partition / T::SWIZZLE_N_PARTITIONS;
            block_n = (partition % T::SWIZZLE_N_PARTITIONS) * n_per_partition + local % n_per_partition;
        } else {
            block_m = grid_m - 1;
            block_n = linear - full_pairs;
        }
    } else if (grid_n <= T::SWIZZLE_MAX_GRID_N && total > T::SWIZZLE_MIN_TILES) {
        const unsigned partitions = ((grid_n <= T::SWIZZLE_SMALL_GRID_N && total < T::SWIZZLE_SMALL_TILES) ||
                                     grid_m <= T::SWIZZLE_SHORT_GRID_M) ? T::SWIZZLE_LARGE_PARTITIONS : T::SWIZZLE_N_PARTITIONS;
        if (total % partitions == 0u) {
            const unsigned logical = (linear % partitions) * (total / partitions) + linear / partitions;
            block_m = logical / grid_n;
            block_n = logical % grid_n;
        }
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch_id = block_id_z();
    const int loops = kargs.k / T::B_K;

    // Matrix and scale global-memory views.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);

    // Matrix and scale layouts: global -> LDS -> registers.
    const int producer_row = lane_id / T::A_K_VECTORS;
    const int producer_xor = (producer_row & 1) | ((producer_row & 2) << 1);
    const auto u_ga = make_layout_ga_scale<T>(lane_id ^ producer_xor, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_ra = layout_9023::make_layout_ra_scale<T>(lane_id, wave_id_m);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_gsfa = layout_9023::make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);

    // Matrix and scale LDS.
    alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
    auto s_sfa = make_smem(reinterpret_cast<u16_t*>(smem_matrix + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

    // MMA and register fragments.
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

    // C fragments are written directly to global memory.
    const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
    D_SF_PACK v_sfa;
    D_SF_PACK v_sfb;

    // Matrix, scale panel and global C offsets.
    auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
    auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
    auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
    auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
    auto ssfa_offset = [&](int tile_k) { return (tile_k & (T::SCALE_PANEL - 1)) * T::SFA_ROWS_PER_REPEAT; };
    auto ssfb_offset = [&](int tile_k) { return tile_k & (T::SCALE_PANEL - 1); };
    auto c_offset = [&](auto c_i) { return gc_offsets[decltype(c_i)::value]; };

    // A/B global memory -> LDS ring stage.
    auto issue_matrix_prefetch = [&](int stage, int tile_k) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
    };

    const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);

    // Scale-panel publication and packed register reads.
    auto load_scale_panel = [&](int begin) {
        const int group = thread_id_x() / T::SFA_PRODUCERS_PER_GROUP;
        const int pair_row = (thread_id_x() % T::SFA_PRODUCERS_PER_GROUP) * T::VEC_SCALE_A;
        if (group < T::SCALE_PANEL && begin + group < loops) {
            const int offset = (begin + group) * kargs.stride_sfa + pair_row;
            const auto low = __builtin_bit_cast(ScaleWords, load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[0] + gsfa_offset(begin)));
            const auto high = __builtin_bit_cast(ScaleWords, load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[1] + gsfa_offset(begin)));
            static_for<2>([&](auto c) {
                constexpr int chunk = decltype(c)::value;
                ScaleWords packed;
                static_for<2>([&](auto w) {
                    constexpr int word = decltype(w)::value, source = chunk * 2 + word;
                    packed[word * 2] = __builtin_amdgcn_perm(high[source], low[source], 0x05010400u);
                    packed[word * 2 + 1] = __builtin_amdgcn_perm(high[source], low[source], 0x07030602u);
                });
                store<8>(s_sfa, __builtin_bit_cast(vector_t<u16_t, 8>, packed),
                         group * T::SFA_ROWS_PER_REPEAT + pair_row + chunk * 8);
            });
        }
        if (thread_id_x() < T::SCALE_PANEL && begin + thread_id_x() < loops) {
            const int group = thread_id_x();
            const D_SF_PACK raw = load<1>(g_sfb, begin + group)[0];
            store<1>(s_sfb, vector_t<D_SF_PACK, 1>{raw * 0x01010101u}, group);
        }
    };
    auto read_scale_a = [&](int tile) {
        return static_cast<D_SF_PACK>(load<1>(s_sfa,
            (tile & (T::SCALE_PANEL - 1)) * T::SFA_ROWS_PER_REPEAT + wave_id_m * T::W_M + lane_id % T::W_M)[0]);
    };
    auto read_scale_b = [&](int tile) {
        return load<1>(s_sfb, tile & (T::SCALE_PANEL - 1))[0];
    };
    // Matrix LDS -> VGPR, one MFMA operand fragment at a time.
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
                                        number<m_repeat>{}, number<n_repeat>{});
        set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
    };
    auto store_output_fragment = [&](auto c_i) {
        constexpr int c_index = decltype(c_i)::value;
        const auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
        store<T::VEC_C>(g_c, cast<D_C>(c), c_offset(c_i));
    };

    int stage = 0;

    // Prologue: seed the matrix ring and the first scale panel.
    static_for<T::PREFETCH_DISTANCE>([&](auto stage_i) {
        if (decltype(stage_i)::value < loops)
            issue_matrix_prefetch(decltype(stage_i)::value, decltype(stage_i)::value);
    });
    load_scale_panel(0);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    v_sfa = read_scale_a(0);
    v_sfb = read_scale_b(0);
    static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, 0); });
    static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, 0); });
    s_waitcnt_lgkmcnt(0_I);

    // Advance each non-final K group, waiting only for the real queued matrix tiles.
#pragma clang loop unroll_count(T::NUM_STAGES)
    for (int tile_k = 0; tile_k + 1 < loops; ++tile_k) {
        const int next_stage = stage == T::NUM_STAGES - 1 ? 0 : stage + 1;
        const int future_stage = stage == 0 ? T::NUM_STAGES - 1 : stage - 1;
        if (((tile_k + 1) & (T::SCALE_PANEL - 1)) == 0) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            load_scale_panel(tile_k + 1);
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        mma_scale_fragment(number<0>{}, number<0>{});
        if (tile_k + T::PREFETCH_DISTANCE < loops)
            issue_matrix_prefetch(future_stage, tile_k + T::PREFETCH_DISTANCE);
        const D_SF_PACK v_sfa_next = read_scale_a(tile_k + 1);
        static_for<T::E_N - 1>([&](auto n_i) {
            mma_scale_fragment(number<0>{}, number<decltype(n_i)::value + 1>{});
        });
        if (tile_k + T::PREFETCH_DISTANCE < loops)
            s_waitcnt_vmcnt(number<T::VMEM_STEADY_WAIT>{});
        else if (T::NUM_STAGES == 4 && tile_k + 2 < loops)
            s_waitcnt_vmcnt(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
        else
            s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        // Replace each operand immediately after its final current use.
        load_a_fragment(number<0>{}, next_stage);
        D_SF_PACK v_sfb_next;
        static_for<T::E_N>([&](auto n_i) {
            constexpr int n_repeat = decltype(n_i)::value;
            mma_scale_fragment(number<1>{}, n_i);
            load_b_fragment(n_i, next_stage);
            if constexpr (n_repeat == 0) v_sfb_next = read_scale_b(tile_k + 1);
        });
        load_a_fragment(number<1>{}, next_stage);
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
        stage = next_stage;
    }

    // Epilogue: finish the final K tile before direct BF16 writeback.
    static_for<T::E_M>([&](auto m_i) {
        static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
    });
    static_for<T::E_M * T::E_N>([&](auto c_i) { store_output_fragment(c_i); });
}
#endif
