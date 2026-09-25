// SPDX-License-Identifier: Apache-2.0
// 64x64 pipeline: four LDS stages, K+3 prefetch and one A/B register buffer.
#pragma once

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_64x64_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_bpreshuffle_64x64_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void gemm_a8w8_mxfp8_bpreshuffle_64x64_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = std::conditional_t<T::SPLIT_K == 1, opus::bf16_t, opus::fp32_t>;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;
    using ScaleWords = opus::vector_t<D_SF_PACK, 4>;

    static_assert(T::B_M == 64 && T::B_N == 64);
    static_assert(T::NUM_STAGES == 4 && T::REG_BUFFERS == 1 && T::SCALE_PANEL == 64);
    static_assert(T::SPLIT_K == 1, "The registered bpreshuffle ABI writes BF16 directly");

    // Tile and thread coordinates.
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    int tile_m = block_id_y();
    int tile_n = block_id_x();
    if constexpr (T::SPLIT_K == 1) {
        // These seven measured N768/K7168 shapes have twelve N tiles. Split
        // the full M-row pairs into 2x4 spatial partitions, interleaved in
        // groups of eight CTAs. An odd final M row is compact, without padded
        // or empty workgroups that would change the dispatch's occupancy tail.
        if (kargs.n == 768 && kargs.k == 7168 && kargs.m >= 1024 && kargs.m <= 1408) {
            constexpr int n_tiles = 12;
            const int m_tiles = kargs.m / T::B_M;
            const int full_pair_tiles = (m_tiles & ~1) * n_tiles;
            const int linear = tile_m * n_tiles + tile_n;
            if (linear < full_pair_tiles) {
                const int partition = linear & 7;
                const int local = linear >> 3;
                tile_m = (local / 3) * 2 + partition / 4;
                tile_n = (partition % 4) * 3 + local % 3;
            } else {
                tile_m = m_tiles - 1;
                tile_n = linear - full_pair_tiles;
            }
        }
    }
    const int row = tile_m * T::B_M;
    const int col = tile_n * T::B_N;

    // Each Split-K workgroup owns a contiguous, whole-K128 interval.
    // The ordinary BF16 instantiation eliminates this partitioning at compile time.
    const int total_loops = kargs.k / T::B_K;
    int split = 0;
    int k_begin = 0;
    int loops = total_loops;
    if constexpr (T::SPLIT_K > 1) {
        const int splits = total_loops < T::SPLIT_K ? total_loops : T::SPLIT_K;
        split = block_id_z();
        k_begin = total_loops * split / splits;
        loops = total_loops * (split + 1) / splits - k_begin;
    }

    // Global-memory views.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + row * kargs.stride_c + col + split * kargs.m * kargs.stride_c);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb));

    // LDS: A[4 stages] | B[4 stages] | packed A scales | replicated B scales.
    alignas(16) __shared__ char smem[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem + T::NUM_STAGES * T::A_STAGE));
    auto s_sfa = make_smem(reinterpret_cast<opus::u16_t*>(smem + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

    // Matrix layouts: global -> LDS -> registers.
    // Permute K16 vectors inside each eight-lane, 128-byte producer row.
    // The global byte set is unchanged; consumer row bits now select different
    // LDS banks. Bit 1 is supplied by the existing 32-byte inter-wave padding.
    const int producer_row = lane_id / (T::B_K / T::VEC_A);
    const int producer_xor = (producer_row & 1) | ((producer_row & 2) << 1);
    const auto u_ga = make_layout_ga_scale<T>(lane_id ^ producer_xor,
        wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);

    // MMA and register fragments.
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(
        seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{},
        seq<T::W_M, T::W_N, T::W_K>{},
        mfma_adaptor_swap_ab{});
    using BaseMMA = typename decltype(mma)::MMA;
    using AFragment = typename BaseMMA::vtype_a;
    using BFragment = typename BaseMMA::vtype_b;
    using AccFragment = typename BaseMMA::vtype_c;
    opus::array<AFragment, T::E_M> v_a;
    opus::array<BFragment, T::E_N> v_b;
    opus::array<AccFragment, T::E_M * T::E_N> v_c{};
    auto ga_offset = [&](int tile_k) { return (k_begin + tile_k) * T::B_K; };
    auto gb_offset = [&](int tile_k) { return (k_begin + tile_k) * T::B_K * T::W_N; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
    auto a_register_offsets = [&](int matrix_stage) {
        opus::array<int, T::E_M * T::A_CHUNKS_PER_FRAGMENT> offsets;
        const int lane_m = lane_id % T::W_M;
        const int consumer_row = lane_m / T::T_M;
        const int consumer_xor = (consumer_row & 1) | ((consumer_row & 2) << 1);
        static_for<T::E_M>([&](auto m_i) {
            constexpr int m_repeat = decltype(m_i)::value;
            const int segment = (m_repeat * T::T_M + wave_id_m) * T::T_M + lane_m % T::T_M;
            const int base = sa_offset(matrix_stage)
                + segment * (T::LDS_WAVE_BYTES + T::LDS_PADDING_BYTES)
                + consumer_row * T::B_K;
            static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto chunk_i) {
                constexpr int chunk = decltype(chunk_i)::value;
                const int logical_k = chunk * (T::WARP_SIZE / T::W_M) + lane_id / T::W_M;
                offsets[m_repeat * T::A_CHUNKS_PER_FRAGMENT + chunk]
                    = base + (logical_k ^ consumer_xor) * T::VEC_A;
            });
        });
        return offsets;
    };
    auto issue_matrix_prefetch = [&](int matrix_stage, int tile_k) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(matrix_stage), ga_offset(tile_k));
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(matrix_stage), gb_offset(tile_k));
    };
    auto load_scale_panel = [&](int panel_begin) {
        // Pair M-repeat 0/1 once when publishing the panel. Two 16-byte
        // global reads produce sixteen packed u16 scales without expansion.
        constexpr int rows_per_repeat = T::SFA_ROWS_PER_REPEAT;
        constexpr int producers_per_group = T::SFA_PRODUCERS_PER_GROUP;
        const int group = thread_id_x() / producers_per_group;
        const int pair_row = (thread_id_x() % producers_per_group) * T::VEC_SCALE_A;
        if (group < T::SCALE_PANEL && panel_begin + group < loops) {
            const int offset = (k_begin + panel_begin + group) * kargs.stride_sfa + row + pair_row;
            const auto low = __builtin_bit_cast(ScaleWords, load<T::VEC_SCALE_A>(g_sfa, offset));
            const auto high = __builtin_bit_cast(ScaleWords,
                load<T::VEC_SCALE_A>(g_sfa, offset + rows_per_repeat));
            static_for<2>([&](auto chunk_i) {
                constexpr int chunk = decltype(chunk_i)::value;
                ScaleWords packed;
                static_for<2>([&](auto word_i) {
                    constexpr int word = decltype(word_i)::value;
                    constexpr int source_word = chunk * 2 + word;
                    // Interleave the two repeat bytes for each of four rows.
                    packed[word * 2] = __builtin_amdgcn_perm(high[source_word], low[source_word], 0x05010400u);
                    packed[word * 2 + 1] = __builtin_amdgcn_perm(high[source_word], low[source_word], 0x07030602u);
                });
                store<8>(s_sfa, __builtin_bit_cast(opus::vector_t<opus::u16_t, 8>, packed),
                         group * rows_per_repeat + pair_row + chunk * 8);
            });
        }
        if (thread_id_x() < T::SCALE_PANEL && panel_begin + thread_id_x() < loops) {
            const int group = thread_id_x();
            const D_SF_PACK raw = load<1>(g_sfb, (col / T::GROUP_N) * kargs.stride_sfb + k_begin + panel_begin + group)[0];
            store<1>(s_sfb, opus::vector_t<D_SF_PACK, 1>{raw * 0x01010101u}, group);
        }
    };
    auto read_scale_a = [&](int tile) {
        const int offset = (tile & (T::SCALE_PANEL - 1)) * T::SFA_ROWS_PER_REPEAT
                         + wave_id_m * T::W_M + lane_id % T::W_M;
        return static_cast<D_SF_PACK>(load<1>(s_sfa, offset)[0]);
    };
    auto read_scale_b = [&](int tile) {
        return load<1>(s_sfb, tile & (T::SCALE_PANEL - 1))[0];
    };

    // The current tile's scales are already in registers. Refill only when
    // the next tile crosses a scale-panel boundary. K7168 has 56 groups,
    // so the seven target shapes use the initial panel for the entire GEMM.
    auto refill_scale_panel = [&](int next_tile) {
        if ((next_tile & (T::SCALE_PANEL - 1)) == 0) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            load_scale_panel(next_tile);
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
    };

    int stage = 0;
    int tile = 0;
    D_SF_PACK v_sfa;
    D_SF_PACK v_sfb;
    D_SF_PACK v_sfa_next;
    D_SF_PACK v_sfb_next;

    // ===== Prologue =====
    // Prime LDS and load K0 into the one A/B register buffer before the loop.
    // Matrix prefetch and scale publication share the initial wait/barrier.
    issue_matrix_prefetch(0, 0);
    if (loops > 1) issue_matrix_prefetch(1, 1);
    if (loops > 2) issue_matrix_prefetch(2, 2);
    load_scale_panel(0);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    v_sfa = read_scale_a(0);
    v_sfb = read_scale_b(0);
    static_for<T::E_M>([&](auto i) {
        constexpr int index = decltype(i)::value;
        v_a[index] = load_operand_fragment_staged<T::VEC_A, index * T::A_CHUNKS_PER_FRAGMENT, AFragment>(
            s_a, a_register_offsets(stage));
    });
    static_for<T::E_N>([&](auto i) {
        constexpr int index = decltype(i)::value;
        v_b[index] = load_operand_fragment_staged<T::VEC_B, index * T::B_CHUNKS_PER_FRAGMENT, BFragment>(
            s_b, layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage)));
    });
    s_waitcnt_lgkmcnt(0_I);

    // ===== Main loop =====
    // Enter with K in registers. Compute K while issuing K+3 to LDS and
    // replacing each operand fragment with K+1 after its last current use.
    // Keep this sequence explicit: it is the measured instruction schedule.
    // C fragment order is A0*B0, A0*B1, A1*B0, A1*B1.
#pragma unroll 4
    for (; tile + T::PREFETCH_DISTANCE < loops; ++tile) {
        const int k_tile = tile;
        const int next_stage = (stage + 1) & (T::NUM_STAGES - 1);
        const int future_stage = (stage + T::PREFETCH_DISTANCE) & (T::NUM_STAGES - 1);
        refill_scale_panel(k_tile + 1);

        mma_scale_group<T, 0, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        issue_matrix_prefetch(future_stage, k_tile + T::PREFETCH_DISTANCE);
        v_sfa_next = read_scale_a(k_tile + 1);
        mma_scale_group<T, 1, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        // Finish K+1 DMA; the eight K+2/K+3 requests may remain in flight.
        // Retire this wave's LDS reads before publishing data across the WG.
        s_waitcnt_vmcnt(number<T::VMEM_STEADY_WAIT>{});
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();

        // Both N fragments have consumed v_a[0]; v_a[1] is still in use.
        v_a[0] = load_operand_fragment_staged<T::VEC_A, 0, AFragment>(
            s_a, a_register_offsets(next_stage));
        mma_scale_group<T, 2, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        // Both M fragments have consumed v_b[0]; v_b[1] is still in use.
        v_b[0] = load_operand_fragment_staged<T::VEC_B, 0, BFragment>(
            s_b, layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage)));
        v_sfb_next = read_scale_b(k_tile + 1);
        mma_scale_group<T, 3, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        v_a[1] = load_operand_fragment_staged<T::VEC_A, T::A_CHUNKS_PER_FRAGMENT, AFragment>(
            s_a, a_register_offsets(next_stage));
        v_b[1] = load_operand_fragment_staged<T::VEC_B, T::B_CHUNKS_PER_FRAGMENT, BFragment>(
            s_b, layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage)));
        // Let operand dependencies place waits at the first consumers.
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
        stage = next_stage;
    }

    // ===== Drain the prefetched K tiles =====
    // Drain up to two queued tiles with compile-time VMEM counts (4, then 0).
    // Each step loads the next operands; the final K tile only computes.
    auto drain_tile = [&](auto pending_vmem) {
        const int next_stage = (stage + 1) & (T::NUM_STAGES - 1);
        refill_scale_panel(tile + 1);
        mma_scale_group<T, 0, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        v_sfa_next = read_scale_a(tile + 1);
        mma_scale_group<T, 1, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        s_waitcnt_vmcnt(pending_vmem);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        v_a[0] = load_operand_fragment_staged<T::VEC_A, 0, AFragment>(
            s_a, a_register_offsets(next_stage));
        mma_scale_group<T, 2, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        v_b[0] = load_operand_fragment_staged<T::VEC_B, 0, BFragment>(
            s_b, layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage)));
        v_sfb_next = read_scale_b(tile + 1);
        mma_scale_group<T, 3, 1>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
        v_a[1] = load_operand_fragment_staged<T::VEC_A, T::A_CHUNKS_PER_FRAGMENT, AFragment>(
            s_a, a_register_offsets(next_stage));
        v_b[1] = load_operand_fragment_staged<T::VEC_B, T::B_CHUNKS_PER_FRAGMENT, BFragment>(
            s_b, layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage)));
        s_waitcnt_lgkmcnt(0_I);
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
        stage = next_stage;
        ++tile;
    };
    if (tile + 2 < loops) drain_tile(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
    if (tile + 1 < loops) drain_tile(0_I);

    // Final K tile: all operands are already in registers.
    mma_scale_group<T, 0, 2>(mma, v_a, v_b, v_c, v_sfa, v_sfb);
    mma_scale_group<T, 2, 2>(mma, v_a, v_b, v_c, v_sfa, v_sfb);

    // ===== Convert and store the output =====
    const auto p_coord = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord);
    const auto offsets = layout_to_offsets<T::VEC_C>(u_gc);
    static_for<T::E_M * T::E_N>([&](auto i) {
        constexpr int index = decltype(i)::value;
        store<T::VEC_C>(g_c, cast<D_C>(v_c[index]), offsets[index]);
    });
}
#endif
