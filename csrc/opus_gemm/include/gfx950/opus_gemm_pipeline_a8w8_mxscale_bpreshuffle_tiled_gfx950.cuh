#pragma once
#include "opus_gemm_mxscale_bpreshuffle_tiled_layout_gfx950.cuh"

namespace opus_bpreshuffle_tiled {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
template<class Traits>
__device__ __forceinline__ void main(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
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

    unsigned raw_sfb_lo, raw_sfb_hi;
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
    auto issue_sfb_panel = [&](int panel_k_begin) {
        const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
        if (k_group < scale_k_groups) {
            asm volatile("buffer_load_ubyte %0, %1, %2, 0 offen" : "=v"(raw_sfb_lo)
                         : "v"(sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin)), "s"(g_sfb.cached_rsrc) : "memory");
            raw_sfb_hi = 0;
            if constexpr (T::SCALE_N_HALVES == 2)
                asm volatile("buffer_load_ubyte %0, %1, %2, 0 offen" : "=v"(raw_sfb_hi)
                             : "v"(sfb_gmem_offsets[1] + gsfb_offset(panel_k_begin)), "s"(g_sfb.cached_rsrc) : "memory");
        }
    };
    auto publish_sfb_panel = [&](int panel_k_begin) {
        const int k_group = panel_k_begin + wave_id * T::WARP_SIZE + lane_id;
        if (k_group < scale_k_groups) {
            unsigned packed_word;
            asm volatile("s_waitcnt vmcnt(0)\n\tv_lshl_or_b32 %0, %2, 8, %1"
                         : "=v"(packed_word) : "v"(raw_sfb_lo), "v"(raw_sfb_hi) : "memory");
            const vector_t<D_SF_PACK, 1> packed{packed_word};
            store<1>(s_sfb, packed, sfb_smem_offsets[0]);
        }
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

    if constexpr (T::FIXED_K == 384) {
        issue_sfb_panel(0);
        __builtin_amdgcn_sched_barrier(0);
        // Spread short-K scale loads across waves without reserving K-specific kernels.
        static_for<T::SFA_K_PASSES>([&](auto i) {
            const int first_k = decltype(i)::value * T::SFA_K_PANEL;
            if (first_k < scale_k_groups) load_sfa_panel(first_k);
        });
        publish_sfb_panel(0);
        __builtin_amdgcn_sched_barrier(0);
    } else {
        // Spread short-K scale loads across waves without reserving K-specific kernels.
        static_for<T::SFA_K_PASSES>([&](auto i) {
            const int first_k = decltype(i)::value * T::SFA_K_PANEL;
            if (first_k < scale_k_groups) load_sfa_panel(first_k);
        });
        load_sfb_panel(0);
        __builtin_amdgcn_sched_barrier(0);
    }
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

template<class Traits>
__device__ __forceinline__ void geometry(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;
    namespace layout_9021 = opus_gemm_4wave_128x128_layout;

    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    const int block_m = block_id_y();
    const int block_n = block_id_x();
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch_id = block_id_z();
    const int loops = (T::FIXED_K ? T::FIXED_K : kargs.k) / T::B_K;

    // Matrix and scale views retain bounded A/C resources for missing M rows.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a,
                        static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col,
                        static_cast<unsigned>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);

    // Matrix and scale layouts: global -> LDS -> registers.
    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_rsfa = opus_gemm_8wave_192x256_layout::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
    const auto u_gsfb = layout_9021::make_layout_gsfb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfb);
    const auto u_ssfb = layout_9021::make_layout_ssfb_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfb = layout_9021::make_layout_rsfb_scale<T>();

    // Matrix and scale LDS; matrix storage is reused for the C epilogue.
    alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem_matrix));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

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
    auto ssfa_offset = [&](int tile_k) { return (tile_k & (T::SCALE_PANEL - 1)) * number<T::B_M>{}; };
    auto ssfb_offset = [&](int tile_k) { return tile_k & (T::SCALE_PANEL - 1); };
    auto c_offset = [&](int row_c, int col_c) { return row_c * T::C_LDS_ROW_STRIDE_ELEMS + col_c; };

    // A/B global memory -> LDS ring stage.
    auto issue_matrix_prefetch = [&](int stage, int tile_k) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile_k));
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile_k));
    };
    const auto sfa_gmem_offsets = opus::transform_tuple([&](auto pass) { return layout_to_offsets<T::VEC_SCALE_A>(layout_9021::make_layout_gsfa_scale<T, decltype(pass)::value>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfa)); }, opus::to_tuple(opus::make_index_seq<T::SFA_PASSES>{}));
    const auto sfa_smem_offsets = opus::transform_tuple([&](auto pass) { return layout_to_offsets<T::VEC_SCALE_A>(layout_9021::make_layout_ssfa_scale<T, decltype(pass)::value>(lane_id, wave_id_m, wave_id_n)); }, opus::to_tuple(opus::make_index_seq<T::SFA_PASSES>{}));
    const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
    const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
    const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
    const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);

    // Issue both raw scale reads before publishing either panel to LDS.
    // Unissued lanes are never published: issue and publish use identical guards.
    array<vector_t<D_SF, T::VEC_SCALE_A>, T::SFA_PASSES> raw_sfa_panel;
    vector_t<D_SF, 1> raw_sfb_panel;
    auto issue_sfa_panel = [&](int panel_k_begin) {
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int smem_offset = opus::get<pass>(sfa_smem_offsets)[0];
            const int local_k_group = smem_offset / T::B_M;
            const int local_row = smem_offset % T::B_M;
            if (smem_offset < T::SFA_BYTES && panel_k_begin + local_k_group < loops) {
                if constexpr (T::SCHEDULE == 1) {
                    raw_sfa_panel[pass] = layout_9021::load_sfa_vector<T>(
                        g_sfa, opus::get<pass>(sfa_gmem_offsets)[0] + gsfa_offset(panel_k_begin),
                        kargs.m - row - local_row);
                } else if (row + local_row < kargs.m) {
                    raw_sfa_panel[pass] = load<T::VEC_SCALE_A>(
                        g_sfa, opus::get<pass>(sfa_gmem_offsets)[0] + gsfa_offset(panel_k_begin));
                } else {
                    static_for<T::VEC_SCALE_A>([&](auto i) { raw_sfa_panel[pass][decltype(i)::value] = 0x7f; });
                }
            }
        });
    };
    auto issue_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            raw_sfb_panel = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin));
        }
    };
    auto publish_sfa_panel = [&](int panel_k_begin) {
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int smem_offset = opus::get<pass>(sfa_smem_offsets)[0];
            const int local_k_group = smem_offset / T::B_M;
            if (smem_offset < T::SFA_BYTES && panel_k_begin + local_k_group < loops) {
                store<T::VEC_SCALE_A>(s_sfa, raw_sfa_panel[pass], smem_offset);
            }
        });
    };
    auto publish_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            store<1>(s_sfb, raw_sfb_panel, sfb_smem_offsets[0]);
        }
    };
    // Scale LDS -> VGPR; pack A repeats after reading LDS.
    auto read_scales = [&](int tile_k, auto& scale_a, auto& scale_b) {
        static_for<T::A_SCALE_PACKS>([&](auto pack_i) { scale_a[decltype(pack_i)::value] = 0; });
        static_for<T::E_M>([&](auto m_i) {
            constexpr int m_repeat = decltype(m_i)::value;
            const D_SF_PACK value = load<1>(s_sfa, rsfa_offsets[m_repeat] + ssfa_offset(tile_k))[0];
            scale_a[m_repeat / 4] |= value << ((m_repeat % 4) * 8);
        });
        static_for<T::B_SCALE_PACKS>([&](auto pack_i) {
            constexpr int pack = decltype(pack_i)::value;
            scale_b[pack] = load<1>(s_sfb, rsfb_offsets[pack] + ssfb_offset(tile_k))[0];
        });
    };
    // Scale A global memory -> VGPR -> LDS, retaining raw E8M0 bytes.
    auto load_sfa_panel = [&](int panel_k_begin) {
        static_for<T::SFA_PASSES>([&](auto pass_i) {
            constexpr int pass = decltype(pass_i)::value;
            const int smem_offset = opus::get<pass>(sfa_smem_offsets)[0];
            const int local_k_group = smem_offset / T::B_M;
            const int local_row = smem_offset % T::B_M;
            if (smem_offset < T::SFA_BYTES && panel_k_begin + local_k_group < loops) {
                vector_t<D_SF, T::VEC_SCALE_A> raw;
                if (row + local_row < kargs.m)
                    raw = load<T::VEC_SCALE_A>(g_sfa, opus::get<pass>(sfa_gmem_offsets)[0] + gsfa_offset(panel_k_begin));
                else
                    static_for<T::VEC_SCALE_A>([&](auto byte_i) { raw[decltype(byte_i)::value] = 0x7f; });
                store<T::VEC_SCALE_A>(s_sfa, raw, smem_offset);
            }
        });
    };
    // Scale B global memory -> VGPR -> LDS, one byte per K128 group.
    auto load_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            store<1>(s_sfb, load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin)),
                     sfb_smem_offsets[0]);
        }
    };

    // Matrix LDS -> registers, one MFMA operand fragment at a time.
    auto load_a_fragment = [&](auto m_i, int stage) {
        layout_9021::load_matrix_fragment<T::VEC_A, ELEM_A>(v_a, s_a, u_ra, m_i, sa_offset(stage));
    };
    auto load_b_fragment = [&](auto n_i, int stage) {
        layout_9021::load_matrix_fragment<T::VEC_B, ELEM_B>(v_b, s_b, u_rb, n_i, sb_offset(stage));
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
                                        static_cast<int>(v_sfa[m_repeat / 4]), static_cast<int>(v_sfb[scale_n_index]),
                                        number<m_repeat % 4>{}, number<0>{});
        set_slice(v_c, c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
    };

    int stage = 0;

    if constexpr (T::SCHEDULE == 1) {
    // Prologue
    issue_matrix_prefetch(0, 0);
    __builtin_amdgcn_sched_barrier(0);
    issue_sfa_panel(0);
    issue_sfb_panel(0);
    __builtin_amdgcn_sched_barrier(0);
    publish_sfa_panel(0);
    publish_sfb_panel(0);
    __builtin_amdgcn_sched_barrier(0);
    if (loops > 1) {
        issue_matrix_prefetch(1, 1);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
    } else {
        s_waitcnt_vmcnt(0_I);
    }
    } else {
    // Prologue
    load_sfa_panel(0);
    load_sfb_panel(0);
    issue_matrix_prefetch(0, 0);
    if (loops > 1)
        issue_matrix_prefetch(1, 1);
    s_waitcnt_vmcnt(0_I);
    }
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(0, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, 0); });
    static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, 0); });
    s_waitcnt_lgkmcnt(0_I);
    if constexpr (T::NUM_STAGES == 2)
        __builtin_amdgcn_s_barrier();

    // Main loop
    for (int tile_k = 0; tile_k + 1 < loops; ++tile_k) {
        const int next_stage = stage + 1 == T::NUM_STAGES ? 0 : stage + 1;
        const int future_stage = T::NUM_STAGES == 2 ? stage : (stage == 0 ? 2 : stage - 1);
        const bool has_future = tile_k + 2 < loops;
        if (has_future)
            issue_matrix_prefetch(future_stage, tile_k + 2);

        if (((tile_k + 1) & (T::SCALE_PANEL - 1)) == 0) {
            if constexpr (T::SCHEDULE == 1) {
                __builtin_amdgcn_sched_barrier(0);
                issue_sfa_panel(tile_k + 1);
                issue_sfb_panel(tile_k + 1);
                __builtin_amdgcn_sched_barrier(0);
                publish_sfa_panel(tile_k + 1);
                publish_sfb_panel(tile_k + 1);
                __builtin_amdgcn_sched_barrier(0);
            } else {
                load_sfa_panel(tile_k + 1);
                load_sfb_panel(tile_k + 1);
            }
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        read_scales(tile_k + 1, v_sfa_next, v_sfb_next);

        static_for<T::E_M>([&](auto m_i) {
            static_for<T::E_N>([&](auto n_i) {
                mma_scale_fragment(m_i, n_i);
                if constexpr (decltype(m_i)::value == T::E_M - 1)
                    load_b_fragment(n_i, next_stage);
            });
            if constexpr (decltype(m_i)::value == 0) {

                __builtin_amdgcn_sched_barrier(0);
                if (has_future)
                    s_waitcnt_vmcnt(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
                else
                    s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
            }
            load_a_fragment(m_i, next_stage);
        });

        if constexpr (T::NUM_STAGES == 2) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
        stage = stage + 1 == T::NUM_STAGES ? 0 : stage + 1;
    }

    // Epilogue
    static_for<T::E_M>([&](auto m_i) {
        static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
    });

    if constexpr (T::NUM_STAGES == 3) {
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }

    // Accumulator -> BF16 -> LDS.
    auto stage_output_fragment = [&](auto c_i) {
        constexpr int c_index = decltype(c_i)::value;
        const auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
        store<T::VEC_C>(s_c, cast<D_C>(c), gc_offsets[c_index]);
    };

    const int output_thread_id = wave_id * T::WARP_SIZE + lane_id;
    auto copy_output_bf16 = [&](auto copy_i) {
        const int linear = output_thread_id * T::VEC_OUTPUT + decltype(copy_i)::value * T::BLOCK_SIZE * T::VEC_OUTPUT;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        const auto value = load<T::VEC_OUTPUT>(s_c, c_offset(output_row, output_col));
        store<T::VEC_OUTPUT>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
    };

    static_for<T::E_M * T::E_N>([&](auto c_i) { stage_output_fragment(c_i); });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    static_for<T::OUTPUT_PASSES>([&](auto copy_i) { copy_output_bf16(copy_i); });
}

template<class Traits>
__device__ __forceinline__ void narrow(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;
    using ScaleWords = vector_t<D_SF_PACK, T::SFA_WORDS>;
    namespace layout_9021 = opus_gemm_4wave_128x128_layout;
    namespace layout_9023 = opus_gemm_4wave_64x128_layout;

    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    const unsigned grid_m = (kargs.m + T::B_M - 1) / T::B_M;
    const unsigned grid_n = kargs.n / T::B_N;
    const unsigned total = grid_m * grid_n;
    unsigned block_m = block_id_y();
    unsigned block_n = block_id_x();
    const unsigned linear = block_m * grid_n + block_n;
    if constexpr (T::BLOCK_GROUP_M > 0) {
        if (grid_n <= T::SWIZZLE_MAX_GRID_N) {
            constexpr unsigned group_m = T::BLOCK_GROUP_M;
            const unsigned first_m = (linear / (group_m * grid_n)) * group_m;
            const unsigned actual_m = grid_m - first_m < group_m ? grid_m - first_m : group_m;
            const unsigned within = linear % (group_m * grid_n);
            block_m = first_m + within % actual_m;
            block_n = within / actual_m;
        }
    } else {
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
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;
    const int batch_id = block_id_z();
    const int loops = T::FIXED_K ? T::FIXED_K / T::B_K : kargs.k / T::B_K;

    // Matrix and scale global-memory views.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a,
                        static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col,
                        static_cast<unsigned>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);

    // Matrix and scale layouts: global -> LDS -> registers.
    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    const auto u_gsfa = layout_9023::make_layout_gsfa_scale<T>(lane_id);
    const auto u_ssfa = layout_9023::make_layout_ssfa_scale<T>(lane_id);
    const auto u_rsfa = layout_9023::make_layout_rsfa_scale<T>(lane_id, wave_id_m);
    const auto u_gsfb = layout_9021::make_layout_gsfb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_sfb);
    const auto u_ssfb = layout_9021::make_layout_ssfb_scale<T>(lane_id, wave_id_m, wave_id_n);
    const auto u_rsfb = layout_9021::make_layout_rsfb_scale<T>();

    // Matrix and scale LDS.
    alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
    auto s_sfa = make_smem(reinterpret_cast<u16_t*>(smem_matrix + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

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
    auto gsfa_offset = [&](int panel_k_begin, int local_k_group) { return (panel_k_begin + local_k_group) * kargs.stride_sfa; };
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
    const auto sfa_smem_offsets = layout_to_offsets<T::VEC_SCALE_PACK_A>(u_ssfa);
    const auto sfb_gmem_offsets = layout_to_offsets<1>(u_gsfb);
    const auto sfb_smem_offsets = layout_to_offsets<1>(u_ssfb);
    const auto rsfa_offsets = layout_to_offsets<1>(u_rsfa);
    const auto rsfb_offsets = layout_to_offsets<1>(u_rsfb);

    // Scale A global memory -> VGPR packing -> LDS u16 pairs.
    auto load_sfa_panel = [&](int panel_k_begin) {
        const int local_k_group = thread_id_x() / T::SFA_PRODUCERS_PER_GROUP;
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            auto load_sfa_vector = [&](int local_row) {
                const auto raw = layout_9021::load_sfa_vector<T>(
                    g_sfa, local_row + gsfa_offset(panel_k_begin, local_k_group),
                    kargs.m - row - local_row);
                return __builtin_bit_cast(ScaleWords, raw);
            };
            const auto low = load_sfa_vector(sfa_gmem_offsets[0]);
            const auto high = load_sfa_vector(sfa_gmem_offsets[1]);
            static_for<T::SFA_PACK_CHUNKS>([&](auto chunk_i) {
                constexpr int chunk = decltype(chunk_i)::value;
                ScaleWords packed;
                static_for<T::SFA_WORDS_PER_CHUNK>([&](auto word_i) {
                    constexpr int word = decltype(word_i)::value;
                    constexpr int source = chunk * T::SFA_WORDS_PER_CHUNK + word;
                    packed[word * T::E_M] = __builtin_amdgcn_perm(high[source], low[source], T::SFA_PACK_PERM_LO);
                    packed[word * T::E_M + 1] = __builtin_amdgcn_perm(high[source], low[source], T::SFA_PACK_PERM_HI);
                });
                store<T::VEC_SCALE_PACK_A>(s_sfa, __builtin_bit_cast(vector_t<u16_t, T::VEC_SCALE_PACK_A>, packed),
                                           sfa_smem_offsets[chunk] + ssfa_offset(local_k_group));
            });
        }
    };
    // Scale B global memory -> VGPR -> LDS, one byte per K128 group.
    auto load_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            store<1>(s_sfb, load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin)),
                     sfb_smem_offsets[0]);
        }
    };
    // Keep the existing u16 panel representation; only split issue and publish.
    // Every publish uses exactly the same guard as its matching issue.
    vector_t<D_SF, T::VEC_SCALE_A> raw_sfa_low;
    vector_t<D_SF, T::VEC_SCALE_A> raw_sfa_high;
    vector_t<D_SF, 1> raw_sfb;
    auto issue_sfa_panel = [&](int panel_k_begin) {
        const int local_k_group = thread_id_x() / T::SFA_PRODUCERS_PER_GROUP;
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            const int low_row = sfa_gmem_offsets[0];
            const int high_row = sfa_gmem_offsets[1];
            raw_sfa_low = layout_9021::load_sfa_vector<T>(
                g_sfa, low_row + gsfa_offset(panel_k_begin, local_k_group),
                kargs.m - row - low_row);
            raw_sfa_high = layout_9021::load_sfa_vector<T>(
                g_sfa, high_row + gsfa_offset(panel_k_begin, local_k_group),
                kargs.m - row - high_row);
        }
    };
    auto issue_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            raw_sfb = load<1>(g_sfb, sfb_gmem_offsets[0] + gsfb_offset(panel_k_begin));
        }
    };
    auto publish_sfa_panel = [&](int panel_k_begin) {
        const int local_k_group = thread_id_x() / T::SFA_PRODUCERS_PER_GROUP;
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            const auto low = __builtin_bit_cast(ScaleWords, raw_sfa_low);
            const auto high = __builtin_bit_cast(ScaleWords, raw_sfa_high);
            static_for<T::SFA_PACK_CHUNKS>([&](auto chunk_i) {
                constexpr int chunk = decltype(chunk_i)::value;
                ScaleWords packed;
                static_for<T::SFA_WORDS_PER_CHUNK>([&](auto word_i) {
                    constexpr int word = decltype(word_i)::value;
                    constexpr int source = chunk * T::SFA_WORDS_PER_CHUNK + word;
                    packed[word * T::E_M] = __builtin_amdgcn_perm(high[source], low[source], T::SFA_PACK_PERM_LO);
                    packed[word * T::E_M + 1] = __builtin_amdgcn_perm(high[source], low[source], T::SFA_PACK_PERM_HI);
                });
                store<T::VEC_SCALE_PACK_A>(s_sfa, __builtin_bit_cast(vector_t<u16_t, T::VEC_SCALE_PACK_A>, packed),
                                           sfa_smem_offsets[chunk] + ssfa_offset(local_k_group));
            });
        }
    };
    auto publish_sfb_panel = [&](int panel_k_begin) {
        const int local_k_group = sfb_smem_offsets[0];
        if (local_k_group < T::SCALE_PANEL && panel_k_begin + local_k_group < loops) {
            store<1>(s_sfb, raw_sfb, sfb_smem_offsets[0]);
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
        layout_9021::load_matrix_fragment<T::VEC_A, ELEM_A>(v_a, s_a, u_ra, m_i, sa_offset(stage));
    };
    auto load_b_fragment = [&](auto n_i, int stage) {
        layout_9021::load_matrix_fragment<T::VEC_B, ELEM_B>(v_b, s_b, u_rb, n_i, sb_offset(stage));
    };
    // Scaled MFMA on one M/N repeat.
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
    auto store_output_fragment = [&](auto c_i) {
        constexpr int c_index = decltype(c_i)::value;
        const auto c = slice(v_c, number<c_index * ELEM_C>{}, number<(c_index + 1) * ELEM_C>{});
        store<T::VEC_C>(g_c, cast<D_C>(c), c_offset(c_i));
    };

    int stage = 0;

    // Prologue
    static_for<T::PREFETCH_DISTANCE>([&](auto stage_i) {
        if (decltype(stage_i)::value < loops)
            issue_matrix_prefetch(decltype(stage_i)::value, decltype(stage_i)::value);
    });
    if constexpr ((T::SCHEDULE == 3 && T::FIXED_K == 0) ||
                  (T::SCHEDULE == 4 && T::FIXED_K == 7168)) {
        issue_sfa_panel(0);
        issue_sfb_panel(0);
        __builtin_amdgcn_sched_barrier(0);
        publish_sfa_panel(0);
        publish_sfb_panel(0);
    } else {
        load_sfa_panel(0);
        load_sfb_panel(0);
    }
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    v_sfa = read_scale_a(0);
    v_sfb = read_scale_b(0);
    if constexpr (T::SCHEDULE == 3) {
        static_for<T::E_M>([&](auto m_i) { load_a_fragment(m_i, 0); });
        static_for<T::E_N>([&](auto n_i) { load_b_fragment(n_i, 0); });
    } else {
        v_a = load<T::VEC_A>(s_a, u_ra);
        v_b = load<T::VEC_B>(s_b, u_rb);
    }
    s_waitcnt_lgkmcnt(0_I);

    // main loop
#pragma clang loop unroll_count(T::SCHEDULE == 3 ? T::NUM_STAGES : 4)
    for (int tile_k = 0; tile_k + 1 < loops; ++tile_k) {
        const int next_stage = stage == T::NUM_STAGES - 1 ? 0 : stage + 1;
        const int future_stage = stage == 0 ? T::NUM_STAGES - 1 : stage - 1;
        if (((tile_k + 1) & (T::SCALE_PANEL - 1)) == 0) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            if constexpr ((T::SCHEDULE == 3 && T::FIXED_K == 0) ||
                  (T::SCHEDULE == 4 && T::FIXED_K == 7168)) {
                issue_sfa_panel(tile_k + 1);
                issue_sfb_panel(tile_k + 1);
                __builtin_amdgcn_sched_barrier(0);
                publish_sfa_panel(tile_k + 1);
                publish_sfb_panel(tile_k + 1);
            } else {
                load_sfa_panel(tile_k + 1);
                load_sfb_panel(tile_k + 1);
            }
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

        D_SF_PACK v_sfb_next;
        if constexpr (T::SCHEDULE == 3) {
            load_a_fragment(number<0>{}, next_stage);
            static_for<T::E_N>([&](auto n_i) {
                constexpr int n_repeat = decltype(n_i)::value;
                mma_scale_fragment(number<1>{}, n_i);
                load_b_fragment(n_i, next_stage);
                if constexpr (n_repeat == 0) v_sfb_next = read_scale_b(tile_k + 1);
            });
            load_a_fragment(number<1>{}, next_stage);
        } else {
        static_for<T::E_N>([&](auto n_i) {
            constexpr int n_repeat = decltype(n_i)::value;
            mma_scale_fragment(number<1>{}, n_i);
            if constexpr (n_repeat == 0) v_sfb_next = read_scale_b(tile_k + 1);
        });
        v_a = load<T::VEC_A>(s_a, u_ra + sa_offset(next_stage));
        v_b = load<T::VEC_B>(s_b, u_rb + sb_offset(next_stage));
        }
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
        stage = next_stage;
    }

    // Epilogue
    static_for<T::E_M>([&](auto m_i) {
        static_for<T::E_N>([&](auto n_i) { mma_scale_fragment(m_i, n_i); });
    });
    static_for<T::E_M * T::E_N>([&](auto c_i) { store_output_fragment(c_i); });
}

#endif
} // namespace opus_bpreshuffle_tiled

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void opus_gemm_mxscale_bpreshuffle_tiled_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void opus_gemm_mxscale_bpreshuffle_tiled_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    if constexpr (Traits::SCHEDULE == 0) opus_bpreshuffle_tiled::main<Traits>(kargs);
    else if constexpr (Traits::SCHEDULE == 1 || Traits::SCHEDULE == 2)
        opus_bpreshuffle_tiled::geometry<Traits>(kargs);
    else opus_bpreshuffle_tiled::narrow<Traits>(kargs);
}
#endif
