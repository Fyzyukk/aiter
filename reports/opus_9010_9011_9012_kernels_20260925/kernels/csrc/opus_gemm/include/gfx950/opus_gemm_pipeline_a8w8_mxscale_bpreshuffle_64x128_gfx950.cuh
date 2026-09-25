// SPDX-License-Identifier: Apache-2.0
// Kid 9011: three matrix LDS stages and one A/B register buffer.
// Use bulk operand loads for small one-panel grids; otherwise replace each
// fragment after its last MFMA use. Both schedules share the same resources.
// Matrix layout/pin helpers retain the imported Fyzyukk/gcnasm implementation
// and Apache-2.0 license in ../../licenses/.
#pragma once

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_64x128_gfx950.cuh"


#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_bpreshuffle_64x128_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void gemm_a8w8_mxfp8_bpreshuffle_64x128_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;

    using T = opus::remove_cvref_t<Traits>;
    using D_A = opus::fp8_t;
    using D_B = opus::fp8_t;
    using D_C = opus::bf16_t;
    using D_ACC = opus::fp32_t;
    using D_SF = unsigned char;
    using D_SF_PACK = unsigned int;

    static_assert(T::B_M == 64 && T::B_N == 128);
    static_assert(T::NUM_STAGES == 3 && T::REG_BUFFERS == 1 && T::SCALE_PANEL == 64);

    // Tile and thread coordinates.
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    unsigned int block_m = block_id_y();
    unsigned int block_n = block_id_x();
    const unsigned int grid_m = kargs.m / T::B_M;
    const unsigned int grid_n = kargs.n / T::B_N;
    // All CTAs select the same partition count from the grid dimensions.
    // Small grids and selected partitions with a short tail stay linear.
    if (grid_n <= 16u && grid_m * grid_n > 256u) {
        const unsigned int linear = block_m * grid_n + block_n;
        const unsigned int total = grid_m * grid_n;
        unsigned int logical = linear;
        bool remap = false;
        if ((grid_n <= 8u && total < 768u) || grid_m <= 24u) {
            if ((total & 15u) == 0u) {
                logical = (linear & 15u) * (total >> 4) + (linear >> 4);
                remap = true;
            }
        } else {
            if ((total & 3u) == 0u) {
                logical = (linear & 3u) * (total >> 2) + (linear >> 2);
                remap = true;
            }
        }
        if (remap) {
            block_m = logical / grid_n;
            block_n = logical % grid_n;
        }
    }
    const int row = block_m * T::B_M;
    const int col = block_n * T::B_N;

    // Global-memory views.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + row * kargs.stride_c + col);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb));

    // Matrix and scale LDS.
    alignas(16) __shared__ char smem[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem + T::NUM_STAGES * T::A_STAGE));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

    // Matrix layouts: global -> LDS -> registers.
    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_ra = make_layout_ra_scale<T>(lane_id, wave_id_m);
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
    using AChunk = opus::vector_t<D_A, T::VEC_A>;
    using BChunk = opus::vector_t<D_B, T::VEC_B>;
    opus::array<AFragment, T::E_M> v_a;
    opus::array<BFragment, T::E_N> v_b;
    using CTile = pinned_accumulator_array<AccFragment, T::E_M * T::E_N, T::C_AGPR_BASE>;
    CTile c00{};
    constexpr int A_DWORDS_PER_FRAGMENT = sizeof(AFragment) / sizeof(opus::u32_t);
    constexpr int B_DWORDS_PER_FRAGMENT = sizeof(BFragment) / sizeof(opus::u32_t);
    auto* a_chunks = reinterpret_cast<AChunk*>(&v_a);
    auto* b_chunks = reinterpret_cast<BChunk*>(&v_b);
    auto ga_offset = [&](int tile_k) { return tile_k * T::B_K; };
    auto gb_offset = [&](int tile_k) { return tile_k * T::B_K * T::W_N; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
    auto issue_matrix_prefetch = [&](int matrix_stage, int tile_k) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(matrix_stage), ga_offset(tile_k));
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(matrix_stage), gb_offset(tile_k));
    };
    const int loops = kargs.k / T::B_K;
    auto load_scale_panel = [&](int panel_begin) {
        static_for<T::SFA_PASSES>([&](auto i) {
            constexpr int pass = decltype(i)::value;
            const int index = (thread_id_x() + pass * T::BLOCK_SIZE) * T::VEC_SCALE_A;
            const int group = index / T::B_M;
            if (index < T::SFA_BYTES && panel_begin + group < loops) {
                const auto values = load<T::VEC_SCALE_A>(g_sfa, (panel_begin + group) * kargs.stride_sfa + row + index % T::B_M);
                store<T::VEC_SCALE_A>(s_sfa, values, index);
            }
        });
        if (thread_id_x() < T::SCALE_PANEL && panel_begin + thread_id_x() < loops) {
            const int group = thread_id_x();
            store<1>(s_sfb, load<1>(g_sfb, (col / T::GROUP_N) * kargs.stride_sfb + panel_begin + group), group);
        }
    };
    auto read_scale_a = [&](int tile) {
        D_SF_PACK packed = 0;
        static_for<T::E_M>([&](auto i) {
            constexpr int repeat = decltype(i)::value;
            const int offset = (tile & (T::SCALE_PANEL - 1)) * T::B_M + wave_id_m * T::W_M + lane_id % T::W_M + repeat * T::T_M * T::W_M;
            packed |= static_cast<D_SF_PACK>(load<1>(s_sfa, offset)[0]) << (repeat * 8);
        });
        return packed;
    };
    auto read_scale_b = [&](int tile) {
        return static_cast<D_SF_PACK>(load<1>(s_sfb, tile & (T::SCALE_PANEL - 1))[0]) * 0x01010101u;
    };

    // The current tile's scales are already in registers. Refill only when
    // the next tile crosses a scale-panel boundary, as in the 9000 pipeline.
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

    // Common prologue: fill the scale panel and prime K0/K1 in LDS.
    load_scale_panel(0);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    issue_matrix_prefetch(0, 0);
    if (loops > 1) issue_matrix_prefetch(1, 1);
    // The bulk schedule preserves the measured baseline early-prefetch
    // window. This once-per-CTA condition is outside both hot loops.
    const bool use_bulk_schedule =
        static_cast<unsigned long long>(grid_m) * grid_n <= 256u &&
        loops >= 32 && loops <= T::SCALE_PANEL;
    if (use_bulk_schedule) {
        v_sfa = read_scale_a(0);
        v_sfb = read_scale_b(0);
        auto bulk_step = [&](auto wait_constant, auto do_prefetch, auto has_next) {
            s_waitcnt_vmcnt(wait_constant);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            if constexpr (decltype(do_prefetch)::value) {
                const int future_stage = stage == 0 ? T::NUM_STAGES - 1 : stage - 1;
                issue_matrix_prefetch(future_stage, tile + T::PREFETCH_DISTANCE);
            }
            load_operand_chunks_pinned<T::VEC_A, T::E_M * T::A_CHUNKS_PER_FRAGMENT, T::A_AGPR_BASE>(
                a_chunks, s_a, u_ra + sa_offset(stage));
            load_operand_chunks_pinned<T::VEC_B, T::E_N * T::B_CHUNKS_PER_FRAGMENT, T::B_AGPR_BASE>(
                b_chunks, s_b, u_rb + sb_offset(stage));
            s_waitcnt_lgkmcnt(0_I);
            mma_scale_group<T, 0, T::E_M * T::E_N>(mma, v_a, v_b, c00, v_sfa, v_sfb);
            if constexpr (decltype(has_next)::value) {
                v_sfa = read_scale_a(tile + 1);
                v_sfb = read_scale_b(tile + 1);
            }
        };
        for (; tile + T::PREFETCH_DISTANCE < loops; ++tile) {
            bulk_step(number<T::VMEM_STEADY_WAIT>{}, true_type{}, true_type{});
            stage = stage == T::NUM_STAGES - 1 ? 0 : stage + 1;
        }
        if (tile + 1 < loops) {
            bulk_step(number<T::VMEM_STEADY_WAIT>{}, false_type{}, true_type{});
            ++tile;
            stage = stage == T::NUM_STAGES - 1 ? 0 : stage + 1;
        }
        bulk_step(0_I, false_type{}, false_type{});
    } else {
        if (loops > 1) s_waitcnt_vmcnt(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
        else s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        v_sfa = read_scale_a(0);
        v_sfb = read_scale_b(0);
        load_operand_chunks_pinned<T::VEC_A, T::E_M * T::A_CHUNKS_PER_FRAGMENT, T::A_AGPR_BASE>(
            a_chunks, s_a, u_ra);
        load_operand_chunks_pinned<T::VEC_B, T::E_N * T::B_CHUNKS_PER_FRAGMENT, T::B_AGPR_BASE>(
            b_chunks, s_b, u_rb);
        s_waitcnt_lgkmcnt(0_I);

        auto interleaved_step = [&](auto do_prefetch, auto wait_constant) {
            const int next_stage = stage == T::NUM_STAGES - 1 ? 0 : stage + 1;
            const int future_stage = stage == 0 ? T::NUM_STAGES - 1 : stage - 1;
            refill_scale_panel(tile + 1);
            mma_scale_group<T, 0, 1>(mma, v_a, v_b, c00, v_sfa, v_sfb);
            if constexpr (decltype(do_prefetch)::value) {
                issue_matrix_prefetch(future_stage, tile + T::PREFETCH_DISTANCE);
            }
            mma_scale_group<T, 1, 1>(mma, v_a, v_b, c00, v_sfa, v_sfb);
            v_sfa_next = read_scale_a(tile + 1);
            mma_scale_group<T, 2, 1>(mma, v_a, v_b, c00, v_sfa, v_sfb);
            s_waitcnt_vmcnt(wait_constant);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            mma_scale_group<T, 3, 1>(mma, v_a, v_b, c00, v_sfa, v_sfb);
            load_operand_fragment_pinned<T::VEC_A, 0>(number<T::A_AGPR_BASE>{},
                v_a[0], s_a, layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage)));
            static_for<T::E_N>([&](auto n) {
                constexpr int j = decltype(n)::value;
                mma_scale_group<T, T::E_N + j, 1>(mma, v_a, v_b, c00, v_sfa, v_sfb);
                load_operand_fragment_pinned<T::VEC_B, j * T::B_CHUNKS_PER_FRAGMENT>(
                    number<T::B_AGPR_BASE + j * B_DWORDS_PER_FRAGMENT>{},
                    v_b[j], s_b, layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage)));
                if constexpr (j == 0) v_sfb_next = read_scale_b(tile + 1);
            });
            load_operand_fragment_pinned<T::VEC_A, T::A_CHUNKS_PER_FRAGMENT>(
                number<T::A_AGPR_BASE + A_DWORDS_PER_FRAGMENT>{},
                v_a[1], s_a, layout_to_offsets<T::VEC_A>(u_ra + sa_offset(next_stage)));
            s_waitcnt_lgkmcnt(0_I);
            v_sfa = v_sfa_next;
            v_sfb = v_sfb_next;
            stage = next_stage;
        };
    #pragma unroll 3
        for (; tile + T::PREFETCH_DISTANCE < loops; ++tile) {
            interleaved_step(true_type{}, number<T::VMEM_STEADY_WAIT>{});
        }
        if (tile + 1 < loops) {
            interleaved_step(false_type{}, 0_I);
            ++tile;
        }
        mma_scale_group<T, 0, T::E_M * T::E_N>(mma, v_a, v_b, c00, v_sfa, v_sfb);
    }
    const auto p_coord = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord);
    const auto offsets = layout_to_offsets<T::VEC_C>(u_gc);
    static_for<T::E_M * T::E_N>([&](auto i) {
        constexpr int index = decltype(i)::value;
        store<T::VEC_C>(g_c, cast<D_C>(c00[index]), offsets[index]);
    });
}
#endif
