#pragma once

#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_8wave_192x256_gfx950.cuh"

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

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_scale_8wave_192x256_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void gemm_a8w8_mxfp8_scale_8wave_192x256_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;
    namespace layout_9020 = opus_gemm_8wave_192x256_layout;

    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    int block_m = block_id_y();
    int block_n = block_id_x();
    if (kargs.n / T::B_N >= T::SWIZZLE_MIN_N_TILES && kargs.m > T::B_M) {
        // Keep the M tiles for a small panel of B close in the launch order.
        const int grid_m = 1 + (kargs.m - 1) / T::B_M;
        const int grid_n = kargs.n / T::B_N;
        const int linear = block_id_y() * grid_n + block_id_x();
        const int group_size = grid_m * T::SWIZZLE_GROUP_N;
        const int first_n = (linear / group_size) * T::SWIZZLE_GROUP_N;
        const int remaining_n = grid_n - first_n;
        const int actual_n = remaining_n < T::SWIZZLE_GROUP_N ? remaining_n : T::SWIZZLE_GROUP_N;
        const int within_group = linear % group_size;
        block_m = within_group / actual_n;
        block_n = first_n + within_group % actual_n;
    } else if (kargs.n <= T::SWIZZLE_MAX_N && kargs.m >= T::SWIZZLE_MIN_M) {
        const int grid_m = 1 + (kargs.m - 1) / T::B_M;
        const int grid_n = kargs.n / T::B_N;
        const int linear = block_id_y() * grid_n + block_id_x();
        const int group_size = T::SWIZZLE_GROUP_M * grid_n;
        const int first_m = (linear / group_size) * T::SWIZZLE_GROUP_M;
        const int remaining_m = grid_m - first_m;
        const int actual_m = remaining_m < T::SWIZZLE_GROUP_M ? remaining_m : T::SWIZZLE_GROUP_M;
        const int within_group = linear % group_size;
        block_m = first_m + within_group % actual_m;
        block_n = within_group / actual_m;
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch_id = block_id_z();
    const int loops = T::FIXED_K ? T::FIXED_K / T::B_K : kargs.k / T::B_K;

    // Matrix and scale global-memory views.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a, static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col, static_cast<unsigned>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);

    // Matrix and scale layouts: global -> LDS -> registers.
    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_ra = layout_9020::make_layout_ra_scale<T>(lane_id, wave_id_m);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_gsfa = layout_9020::make_layout_gsfa_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa);
    const auto u_ssfa = layout_9020::make_layout_ssfa_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfa = layout_9020::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
    const auto u_gsfb = layout_9020::make_layout_gsfb_scale<T>(lane_id, wave_id, kargs.stride_sfb);
    const auto u_ssfb = layout_9020::make_layout_ssfb_scale<T>(lane_id, wave_id);
    const auto u_rsfb = layout_9020::make_layout_rsfb_scale();

    // Matrix and scale LDS; matrix storage is reused for the C epilogue.
    alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
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

    // C output layout; BF16 fragments are first staged in LDS.
    const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(number<T::C_LDS_ROW_STRIDE_ELEMS>{}, 1_I), p_coord_c);
    const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
    array<D_SF_PACK, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
    array<D_SF_PACK, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};

    // Matrix, scale panel and C staging offsets.
    auto ga_offset = [&](int tile_k) { return tile_k * number<T::B_K>{}; };
    auto gb_offset = [&](int tile_k) { return tile_k * number<T::B_K * T::W_N>{}; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
    auto gsfa_offset = [&](int panel_k_begin) { return panel_k_begin * kargs.stride_sfa; };
    auto gsfb_offset = [&](int panel_k_begin) { return panel_k_begin; };
    auto ssfa_offset = [&](int tile_k) { return tile_k * number<T::B_M>{}; };
    // s_sfb addresses packed uint words, one word per K128 group.
    auto ssfb_offset = [&](int tile_k) { return tile_k; };
    auto c_offset = [&](int row_c, int col_c) { return row_c * T::C_LDS_ROW_STRIDE_ELEMS + col_c; };

    // A/B global memory -> LDS ping-pong stage.
    auto issue_matrix_prefetch = [&](auto stage_i, int tile_k) {
        constexpr int stage = decltype(stage_i)::value;
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
    };
    const int scale_k_groups = T::FIXED_K ? T::FIXED_K / T::GROUP_K : kargs.k / T::GROUP_K;
    const auto sfa_gmem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_gsfa);
    const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_A>(u_ssfa);
    const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
    const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
    const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
    const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);

    // Scale A global memory -> VGPR -> LDS, retaining raw E8M0 bytes.
    auto load_sfa_panel = [&](int panel_k_begin) {
        const int local_k_group = wave_id * T::SFA_K_COLUMNS_PER_WAVE + lane_id / T::SFA_THREADS_PER_GROUP;
        const int k_group = panel_k_begin + local_k_group;
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int local_row = sfa_smem_offsets[pass] - local_k_group * T::B_M;
            const int smem_offset = sfa_smem_offsets[pass] + panel_k_begin * T::B_M;
            if (local_row < T::B_M && smem_offset < T::SFA_BYTES && k_group < scale_k_groups) {
                vector_t<D_SF, T::VEC_SCALE_A> raw;
                if (row + local_row < kargs.m)
                    raw = load<T::VEC_SCALE_A>(g_sfa, sfa_gmem_offsets[pass] + gsfa_offset(panel_k_begin));
                else
                    static_for<T::VEC_SCALE_A>([&](auto byte_i) { raw[decltype(byte_i)::value] = 0x7f; });
                store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);
            }
        });
    };
    // Scale B global memory -> VGPR packing -> LDS.
    auto load_sfb_panel = [&](int panel_k_begin) {
        const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
        if (k_group < scale_k_groups) {
            const unsigned lo = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin))[0];
            unsigned hi = 0;
            if constexpr (T::SCALE_N_HALVES == 2)
                hi = load<1>(g_sfb, sfb_gmem_offsets[1] + gsfb_offset(panel_k_begin))[0];
            // One word per K128 group: low/high bytes select the two N128 halves.
            const vector_t<D_SF_PACK, 1> packed{lo | (hi << 8)};
            store<1>(s_sfb, packed, sfb_smem_offsets[0]);
        }
    };
    // Scale LDS -> VGPR; pack A repeats after reading LDS.
    auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
        static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
        static_for<T::E_M>([&](auto m_i) {
            constexpr int m_repeat = decltype(m_i)::value;
            const unsigned value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
            scale_a[m_repeat / 4] |= value << ((m_repeat % 4) * 8);
        });
        scale_b[0] = load<1>(s_sfb, rsfb_offsets[0] + ssfb_offset(tile_k))[0];
    };
    // Matrix LDS -> VGPR, one MFMA operand fragment at a time.
    auto load_a_fragment = [&](auto m_i, auto stage_i) {
        constexpr int m_repeat = decltype(m_i)::value;
        constexpr int stage = decltype(stage_i)::value;
        const auto offsets = layout_to_offsets<T::VEC_A>(u_ra + sa_offset(stage));
        static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
            constexpr int index = m_repeat * T::A_CHUNKS_PER_FRAGMENT + decltype(chunk_i)::value;
            set_slice(v_a, load<T::VEC_A>(s_a, offsets[index]),
                      number<index * T::VEC_A>{}, number<(index + 1) * T::VEC_A>{});
        });
    };
    auto load_b_fragment = [&](auto n_i, auto stage_i) {
        constexpr int n_repeat = decltype(n_i)::value;
        constexpr int stage = decltype(stage_i)::value;
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
        constexpr int scale_n_index = n_repeat / (T::GROUP_N / (T::T_N * T::W_N));
        const auto a = slice(v_a, number<m_repeat * ELEM_A>{}, number<(m_repeat + 1) * ELEM_A>{});
        const auto b = slice(v_b, number<n_repeat * ELEM_B>{}, number<(n_repeat + 1) * ELEM_B>{});
        auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
        c = typename decltype(mma)::MMA{}(a, b, c,
                                        static_cast<int>(v_sfa[m_repeat / 4]), static_cast<int>(v_sfb[0]),
                                        number<m_repeat % 4>{}, number<scale_n_index>{});
        set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
    };

    auto advance_tile = [&](auto stage_i, int tile_k) {
        constexpr int stage = decltype(stage_i)::value;
        constexpr int next_stage = (stage + 1) % T::NUM_STAGES;
        read_scales(tile_k + 1, v_sfa_next, v_sfb_next);
        // Overlap the first M repeat's eight MFMAs with the pending next matrix tile.
        static_for<T::E_M>([&](auto m_i) {
            static_for<T::E_N>([&](auto n_i) {
                mma_scale_fragment(m_i, n_i);
                if constexpr (decltype(m_i)::value == T::E_M - 1)
                    load_b_fragment(n_i, number<next_stage>{});
            });
            if constexpr (decltype(m_i)::value == 0) {
                __builtin_amdgcn_sched_barrier(0);
                s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                // Publish K+1 and retire all K readers before K+2 reuses the same LDS slot.
                __builtin_amdgcn_s_barrier();
                if (tile_k + 2 < loops)
                    issue_matrix_prefetch(stage_i, tile_k + 2);
                __builtin_amdgcn_sched_barrier(0);
            }
            // Replace A after its last N use and B after its last M use.
            load_a_fragment(m_i, number<next_stage>{});
        });
        // Complete this wave's next-tile reads; the next midpoint barrier protects LDS reuse.
        s_waitcnt_lgkmcnt(0_I);
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    };

    // Prologue
    issue_matrix_prefetch(number<0>{}, 0);
    __builtin_amdgcn_sched_barrier(0);

    // Spread short-K scale loads across waves without reserving K-specific kernels.
    static_for<T::SFA_K_PASSES>([&](auto i) {
        const int first_k = decltype(i)::value * T::SFA_K_PANEL;
        if (first_k < scale_k_groups) load_sfa_panel(first_k);
    });
    load_sfb_panel(0);
    __builtin_amdgcn_sched_barrier(0);
    if (loops > 1) {
        issue_matrix_prefetch(number<1>{}, 1);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<T::MATRIX_VMEM_INSTRUCTIONS>{});
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(0, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, number<0>{}); });
    static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, number<0>{}); });
    s_waitcnt_lgkmcnt(0_I);

    // Main loop
    for (int tile_k = 0; tile_k + 1 < loops; tile_k += T::LOOP_UNROLL) {
        advance_tile(number<0>{}, tile_k);

        if (tile_k + 2 < loops)
            advance_tile(number<1>{}, tile_k + 1);
    }

    // Epilogue
    auto stage_output_fragment = [&](auto c_i) {
        constexpr int c_index = decltype(c_i)::value;
        const auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
        store<T::VEC_C>(s_c, cast<D_C>(c), gc_offsets[c_index]);
    };

    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    static_for<T::E_M>([&](auto m_i) {
        static_for<T::E_N>([&](auto n_i) {
            constexpr int c_index = decltype(m_i)::value * T::E_N + decltype(n_i)::value;
            mma_scale_fragment(m_i, n_i);
            stage_output_fragment(number<c_index>{});
        });
    });
    // Output writeback
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
    auto copy_output_bf16 = [&](int copy_index) {
        const int linear = output_thread_id * T::VEC_OUTPUT + copy_index * T::BLOCK_SIZE * T::VEC_OUTPUT;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
        if (row + output_row < kargs.m)
            store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col,
                     0, opus::number<2>{});
    };
    static_for<T::OUTPUT_PASSES>([&](auto copy_i) { copy_output_bf16(decltype(copy_i)::value); });
}
#endif
