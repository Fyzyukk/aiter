// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "traits_runtime.cuh"

namespace opus_gemm_4wave_64x128_layout {

using opus::operator""_I;

template<class T>
__device__ inline constexpr auto make_layout_ga_scale(int lane_id, int wave_id_m, int wave_id_n, int stride_a) {
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
    const auto u_ga = layout_9023::make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_gsfa = layout_9023::make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa = layout_9023::make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfa = layout_9023::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
    const auto u_gsfb = layout_9023::make_layout_gsfb_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_ssfb = layout_9023::make_layout_ssfb_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfb = layout_9023::make_layout_rsfb_scale();

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
    array<typename decltype(mma)::MMA::vtype_a, T::E_M> v_a;
    array<typename decltype(mma)::MMA::vtype_b, T::E_N> v_b;
    array<typename decltype(mma)::MMA::vtype_c, T::E_M * T::E_N> v_c;
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

    auto a_offsets = [&](int stage) {
        array<int, T::E_M * T::A_CHUNKS_PER_FRAGMENT> offsets;
        const int lane_m = lane_id % T::W_M, consumer_row = lane_m / T::T_M;
        const int consumer_xor = (consumer_row & 1) | ((consumer_row & 2) << 1);
        static_for<T::E_M>([&](auto m) {
            constexpr int repeat = decltype(m)::value;
            const int segment = (repeat * T::T_M + wave_id_m) * T::T_M + lane_m % T::T_M;
            const int base = sa_offset(stage) + segment * (T::smem_linear_wave + T::smem_padding)
                           + consumer_row * T::B_K;
            static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto c) {
                constexpr int chunk = decltype(c)::value;
                const int logical_k = chunk * (T::WARP_SIZE / T::W_M) + lane_id / T::W_M;
                offsets[repeat * T::A_CHUNKS_PER_FRAGMENT + chunk] = base + (logical_k ^ consumer_xor) * T::VEC_A;
            });
        });
        return offsets;
    };
    // A/B global memory -> LDS ring stage.
    auto issue_matrix_prefetch = [&](int stage, int tile_k) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
    };
    const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
    const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_PACK_A>(u_ssfa);
    const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
    const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
    const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
    const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);

    // Scale A/B global memory -> VGPR packing -> LDS.
    auto load_scale_panel = [&](int panel_k_begin) {
        const int local_k_group = thread_id_x() / T::SFA_PRODUCERS_PER_GROUP;
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            const auto low = __builtin_bit_cast(ScaleWords, load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[0] + gsfa_offset(panel_k_begin)));
            const auto high = __builtin_bit_cast(ScaleWords, load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[1] + gsfa_offset(panel_k_begin)));
            static_for<T::SFA_PACK_CHUNKS>([&](auto chunk_i) {
                constexpr int chunk = decltype(chunk_i)::value;
                ScaleWords packed;
                static_for<T::SFA_WORDS_PER_CHUNK>([&](auto word_i) {
                    constexpr int word = decltype(word_i)::value;
                    constexpr int source = chunk * T::SFA_WORDS_PER_CHUNK + word;
                    packed[word * T::E_M] = __builtin_amdgcn_perm(high[source], low[source], T::SFA_PACK_PERM_LO);
                    packed[word * T::E_M + 1] = __builtin_amdgcn_perm(high[source], low[source], T::SFA_PACK_PERM_HI);
                });
                store<T::VEC_SCALE_PACK_A>(s_sfa, __builtin_bit_cast(vector_t<u16_t, T::VEC_SCALE_PACK_A>, packed), sfa_smem_offsets[chunk]);
            });
        }
        if (thread_id_x() < T::SCALE_PANEL && panel_k_begin + thread_id_x() < loops) {
            const D_SF_PACK raw = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin))[0];
            store<1>(s_sfb, vector_t<D_SF_PACK, 1>{raw * T::SFB_REPLICATE}, sfb_smem_offsets[0]);
        }
    };
    // Scale LDS -> VGPR; A and B reads retain their separate scheduling points.
    auto read_scale_a = [&](int tile_k) {
        return static_cast<D_SF_PACK>(load<1>(s_sfa, rsfa_offsets[0] + ssfa_offset(tile_k))[0]);
    };
    auto read_scale_b = [&](int tile_k) {
        return load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
    };
    // Matrix LDS -> VGPR, one MFMA operand fragment at a time.
    auto load_a_fragment = [&](auto m_i, int stage) {
        constexpr int m_repeat = decltype(m_i)::value;
        const auto offsets = a_offsets(stage);
        v_a[m_repeat] = load_operand_fragment_staged<
            T::VEC_A, m_repeat * T::A_CHUNKS_PER_FRAGMENT, typename decltype(mma)::MMA::vtype_a>(s_a, offsets);
    };
    auto load_b_fragment = [&](auto n_i, int stage) {
        constexpr int n_repeat = decltype(n_i)::value;
        const auto offsets = layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage));
        v_b[n_repeat] = load_operand_fragment_staged<
            T::VEC_B, n_repeat * T::B_CHUNKS_PER_FRAGMENT, typename decltype(mma)::MMA::vtype_b>(s_b, offsets);
    };
    auto store_output_fragment = [&](auto c_i) {
        constexpr int c_index = decltype(c_i)::value;
        store<T::VEC_C>(g_c, cast<D_C>(v_c[c_index]), c_offset(c_i));
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
        mma_scale_group<T, 0, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        if (tile_k + T::PREFETCH_DISTANCE < loops)
            issue_matrix_prefetch(future_stage, tile_k + T::PREFETCH_DISTANCE);
        const D_SF_PACK v_sfa_next = read_scale_a(tile_k + 1);
        mma_scale_group<T, 1, T::E_N - 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
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
            mma_scale_group<T, T::E_N + n_repeat, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
            load_b_fragment(n_i, next_stage);
            if constexpr (n_repeat == 0) v_sfb_next = read_scale_b(tile_k + 1);
        });
        load_a_fragment(number<1>{}, next_stage);
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
        stage = next_stage;
    }

    // Epilogue: finish the final K tile before direct BF16 writeback.
    mma_scale_group<T, 0, T::E_M * T::E_N>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
    static_for<T::E_M * T::E_N>([&](auto c_i) { store_output_fragment(c_i); });
}
#endif
