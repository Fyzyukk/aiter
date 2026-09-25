// SPDX-License-Identifier: Apache-2.0
// 64x128: the registered S3/K+2 configuration retains the small-grid bulk
// and interleaved schedules with packed scales and compiler register allocation.
// Imported layout helpers retain their Apache-2.0 provenance in ../../licenses/.
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
    using ScaleWords = opus::vector_t<D_SF_PACK, 4>;
    using PackedScaleA = std::conditional_t<T::E_M == 2, opus::u16_t, opus::u32_t>;

    static_assert(T::B_M == 64 && T::B_N == 128);
    static_assert((T::NUM_STAGES == 2 || T::NUM_STAGES == 3) && T::REG_BUFFERS == 1 && T::SCALE_PANEL == 64);

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
    auto s_sfa = make_smem(reinterpret_cast<PackedScaleA*>(smem + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

    // Matrix layouts: global -> LDS -> registers.
    // Permute K16 vectors within each 128-byte A producer row. The consumer
    // applies the same XOR so adjacent rows use different LDS banks.
    const int producer_row = lane_id / (T::B_K / T::VEC_A);
    const int producer_xor = (producer_row & 1) | ((producer_row & 2) << 1)
                           | (T::NUM_STAGES == 2 ? wave_id_m << 1 : 0);
    const auto u_ga = make_layout_ga_scale<T>(lane_id ^ producer_xor, wave_id_m, wave_id_n, kargs.stride_a);
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
    using CTile = opus::array<AccFragment, T::E_M * T::E_N>;
    CTile c00{};
    auto ga_offset = [&](int tile_k) { return tile_k * T::B_K; };
    auto gb_offset = [&](int tile_k) { return tile_k * T::B_K * T::W_N; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };
    auto a_register_offsets = [&](int matrix_stage) {
        opus::array<int, T::E_M * T::A_CHUNKS_PER_FRAGMENT> offsets;
        const int lane_m = lane_id % T::W_M;
        const int consumer_row = lane_m / T::T_M;
        const int consumer_xor = (consumer_row & 1) | ((consumer_row & 2) << 1)
                               | (T::NUM_STAGES == 2 ? (lane_m % T::T_M) << 1 : 0);
        static_for<T::E_M>([&](auto m_i) {
            constexpr int m_repeat = decltype(m_i)::value;
            const int segment = (m_repeat * T::T_M + wave_id_m) * T::T_M + lane_m % T::T_M;
            const int base = sa_offset(matrix_stage)
                + segment * (T::smem_linear_wave + T::smem_padding)
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
    const int loops = kargs.k / T::B_K;
    // Pack M repeats once per panel; hot-loop A and B reads are one LDS
    // operation each. The packed A panel occupies the original byte count.
    auto load_scale_panel = [&](int panel_begin) {
        constexpr int rows_per_repeat = T::T_M * T::W_M;
        constexpr int producers_per_group = rows_per_repeat / T::VEC_SCALE_A;
        const int group = thread_id_x() / producers_per_group;
        const int pair_row = (thread_id_x() % producers_per_group) * T::VEC_SCALE_A;
        if (group < T::SCALE_PANEL && panel_begin + group < loops) {
            const int offset = (panel_begin + group) * kargs.stride_sfa + row + pair_row;
            opus::array<ScaleWords, T::E_M> words;
            static_for<T::E_M>([&](auto r) {
                constexpr int repeat = decltype(r)::value;
                words[repeat] = __builtin_bit_cast(ScaleWords,
                    load<T::VEC_SCALE_A>(g_sfa, offset + repeat * rows_per_repeat));
            });
            if constexpr (T::E_M == 2) {
                static_for<2>([&](auto ci) {
                    constexpr int chunk = decltype(ci)::value;
                    ScaleWords packed;
                    static_for<2>([&](auto wi) {
                        constexpr int word = decltype(wi)::value;
                        constexpr int source = chunk * 2 + word;
                        packed[word * 2] = __builtin_amdgcn_perm(words[1][source], words[0][source], 0x05010400u);
                        packed[word * 2 + 1] = __builtin_amdgcn_perm(words[1][source], words[0][source], 0x07030602u);
                    });
                    store<8>(s_sfa, __builtin_bit_cast(opus::vector_t<PackedScaleA, 8>, packed),
                             group * rows_per_repeat + pair_row + chunk * 8);
                });
            } else {
                static_assert(T::E_M == 4);
                static_for<4>([&](auto wi) {
                    constexpr int word = decltype(wi)::value;
                    const auto low01 = __builtin_amdgcn_perm(words[1][word], words[0][word], 0x05010400u);
                    const auto high01 = __builtin_amdgcn_perm(words[1][word], words[0][word], 0x07030602u);
                    const auto low23 = __builtin_amdgcn_perm(words[3][word], words[2][word], 0x05010400u);
                    const auto high23 = __builtin_amdgcn_perm(words[3][word], words[2][word], 0x07030602u);
                    ScaleWords packed;
                    packed[0] = __builtin_amdgcn_perm(low23, low01, 0x05040100u);
                    packed[1] = __builtin_amdgcn_perm(low23, low01, 0x07060302u);
                    packed[2] = __builtin_amdgcn_perm(high23, high01, 0x05040100u);
                    packed[3] = __builtin_amdgcn_perm(high23, high01, 0x07060302u);
                    store<4>(s_sfa, packed, group * rows_per_repeat + pair_row + word * 4);
                });
            }
        }
        if (thread_id_x() < T::SCALE_PANEL && panel_begin + thread_id_x() < loops) {
            const int group = thread_id_x();
            const D_SF_PACK raw = load<1>(g_sfb, (col / T::GROUP_N) * kargs.stride_sfb + panel_begin + group)[0];
            store<1>(s_sfb, opus::vector_t<D_SF_PACK, 1>{raw * 0x01010101u}, group);
        }
    };
    auto read_scale_a = [&](int tile) {
        const int offset = (tile & (T::SCALE_PANEL - 1)) * (T::T_M * T::W_M)
                         + wave_id_m * T::W_M + lane_id % T::W_M;
        return static_cast<D_SF_PACK>(load<1>(s_sfa, offset)[0]);
    };
    auto read_scale_b = [&](int tile) {
        return load<1>(s_sfb, tile & (T::SCALE_PANEL - 1))[0];
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
    issue_matrix_prefetch(0, 0);
    if (loops > 1) issue_matrix_prefetch(1, 1);
    load_scale_panel(0);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    // The bulk schedule preserves the measured baseline early-prefetch
    // window. This once-per-CTA condition is outside both hot loops.
    const bool use_bulk_schedule = T::NUM_STAGES == 3 &&
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
                const int future_stage = T::NUM_STAGES == 2 ? stage : (stage == 0 ? T::NUM_STAGES - 1 : stage - 1);
                issue_matrix_prefetch(future_stage, tile + T::PREFETCH_DISTANCE);
            }
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
        v_sfa = read_scale_a(0);
        v_sfb = read_scale_b(0);
        static_for<T::E_M>([&](auto i) {
                constexpr int index = decltype(i)::value;
                v_a[index] = load_operand_fragment_staged<T::VEC_A, index * T::A_CHUNKS_PER_FRAGMENT, AFragment>(
                    s_a, a_register_offsets(0));
            });
        static_for<T::E_N>([&](auto i) {
                constexpr int index = decltype(i)::value;
                v_b[index] = load_operand_fragment_staged<T::VEC_B, index * T::B_CHUNKS_PER_FRAGMENT, BFragment>(
                    s_b, layout_to_offsets<T::VEC_B>(u_rb));
            });
        s_waitcnt_lgkmcnt(0_I);
        if constexpr (T::NUM_STAGES == 2) {
            // S2 next writes K+2 into the slot whose K0 readers just retired.
            __builtin_amdgcn_s_barrier();
        }

        auto interleaved_step = [&](auto do_prefetch, auto wait_constant) {
            const int next_stage = stage == T::NUM_STAGES - 1 ? 0 : stage + 1;
            const int future_stage = T::NUM_STAGES == 2 ? stage : (stage == 0 ? T::NUM_STAGES - 1 : stage - 1);
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
            v_a[0] = load_operand_fragment_staged<T::VEC_A, 0, AFragment>(
                s_a, a_register_offsets(next_stage));
            static_for<T::E_N>([&](auto n) {
                constexpr int j = decltype(n)::value;
                mma_scale_group<T, T::E_N + j, 1>(mma, v_a, v_b, c00, v_sfa, v_sfb);
                v_b[j] = load_operand_fragment_staged<T::VEC_B, j * T::B_CHUNKS_PER_FRAGMENT, BFragment>(
                s_b, layout_to_offsets<T::VEC_B>(u_rb + sb_offset(next_stage)));
                if constexpr (j == 0) v_sfb_next = read_scale_b(tile + 1);
            });
            v_a[1] = load_operand_fragment_staged<T::VEC_A, T::A_CHUNKS_PER_FRAGMENT, AFragment>(
                s_a, a_register_offsets(next_stage));
            if constexpr (T::NUM_STAGES == 2) {
                // All waves must finish K+1 LDS reads before K+3 overwrites it.
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
            }
            v_sfa = v_sfa_next;
            v_sfb = v_sfb_next;
            stage = next_stage;
        };
    #pragma clang loop unroll_count(T::NUM_STAGES)
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
