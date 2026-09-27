// SPDX-License-Identifier: Apache-2.0
// Shared 64x{64,128} runtime-K flow: one priming pass, one advancing loop,
// one final compute/output. No bulk/interleaved choice or separate drain loop.
#pragma once
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_narrow_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_bpreshuffle_narrow_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void gemm_a8w8_mxfp8_bpreshuffle_narrow_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;
    using T = remove_cvref_t<Traits>;
    using ScaleWords = vector_t<u32_t, 4>;
    const int wave = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane = thread_id_x() % T::WARP_SIZE;
    const int wave_m = wave % T::T_M, wave_n = wave / T::T_M;
    const unsigned grid_m = kargs.m / T::B_M, grid_n = kargs.n / T::B_N;
    const unsigned total = grid_m * grid_n;
    unsigned tile_m = block_id_y(), tile_n = block_id_x();
    const unsigned linear = tile_m * grid_n + tile_n;
    // Geometry-only traversal. Compact 64-column tiles interleave pairs of
    // M rows; 128-column tiles retain the measured general partition rule.
    // Neither policy depends on K or an enumerated shape.
    if (T::B_N == 64 && grid_n <= 16u && (grid_n & 3u) == 0u && grid_m >= 16u && grid_m <= 32u) {
        const unsigned n_per_partition = grid_n / 4;
        const unsigned full_pairs = (grid_m & ~1u) * grid_n;
        if (linear < full_pairs) {
            const unsigned partition = linear & 7u, local = linear >> 3;
            tile_m = (local / n_per_partition) * 2 + partition / 4;
            tile_n = (partition % 4) * n_per_partition + local % n_per_partition;
        } else {
            tile_m = grid_m - 1;
            tile_n = linear - full_pairs;
        }
    } else if (grid_n <= 16u && total > 256u) {
        const unsigned partitions = ((grid_n <= 8u && total < 768u) || grid_m <= 24u) ? 16u : 4u;
        if (total % partitions == 0u) {
            const unsigned logical = (linear % partitions) * (total / partitions) + linear / partitions;
            tile_m = logical / grid_n;
            tile_n = logical % grid_n;
        }
    }
    const int row = tile_m * T::B_M, col = tile_n * T::B_N;
    const int loops = kargs.k / T::B_K;
    auto g_a = make_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_a) + row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const fp8_t*>(kargs.ptr_b) + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<bf16_t*>(kargs.ptr_c) + row * kargs.stride_c + col);
    auto g_sfa = make_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const unsigned char*>(kargs.ptr_sfb));
    alignas(16) __shared__ char smem[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<fp8_t*>(smem));
    auto s_b = make_smem(reinterpret_cast<fp8_t*>(smem + T::NUM_STAGES * T::A_STAGE));
    auto s_sfa = make_smem(reinterpret_cast<u16_t*>(smem + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<u32_t*>(smem + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

    const int producer_row = lane / (T::B_K / T::VEC_A);
    const int producer_xor = (producer_row & 1) | ((producer_row & 2) << 1);
    const auto u_ga = make_layout_ga_scale<T>(lane ^ producer_xor, wave_m, wave_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_m, wave_n);
    const auto u_gb = make_layout_gb_scale<T>(lane, wave_m, wave_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_m, wave_n);
    const auto u_rb = make_layout_rb_scale<T>(lane, wave_n);
    auto a_offsets = [&](int stage) {
        array<int, T::E_M * T::A_CHUNKS_PER_FRAGMENT> offsets;
        const int lane_m = lane % T::W_M, consumer_row = lane_m / T::T_M;
        const int consumer_xor = (consumer_row & 1) | ((consumer_row & 2) << 1);
        static_for<T::E_M>([&](auto m) {
            constexpr int repeat = decltype(m)::value;
            const int segment = (repeat * T::T_M + wave_m) * T::T_M + lane_m % T::T_M;
            const int base = stage * T::A_STAGE + segment * (T::smem_linear_wave + T::smem_padding)
                           + consumer_row * T::B_K;
            static_for<T::A_CHUNKS_PER_FRAGMENT>([&](auto c) {
                constexpr int chunk = decltype(c)::value;
                const int logical_k = chunk * (T::WARP_SIZE / T::W_M) + lane / T::W_M;
                offsets[repeat * T::A_CHUNKS_PER_FRAGMENT + chunk] = base + (logical_k ^ consumer_xor) * T::VEC_A;
            });
        });
        return offsets;
    };
    auto prefetch = [&](int stage, int tile) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + stage * T::A_STAGE, tile * T::B_K);
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + stage * T::B_STAGE, tile * T::B_K * T::W_N);
    };
    auto publish_scales = [&](int begin) {
        const int group = thread_id_x() / T::SFA_PRODUCERS_PER_GROUP;
        const int pair_row = (thread_id_x() % T::SFA_PRODUCERS_PER_GROUP) * T::VEC_SCALE_A;
        if (group < T::SCALE_PANEL && begin + group < loops) {
            const int offset = (begin + group) * kargs.stride_sfa + row + pair_row;
            const auto low = __builtin_bit_cast(ScaleWords, load<T::VEC_SCALE_A>(g_sfa, offset));
            const auto high = __builtin_bit_cast(ScaleWords, load<T::VEC_SCALE_A>(g_sfa, offset + T::SFA_ROWS_PER_REPEAT));
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
            const u32_t raw = load<1>(g_sfb, (col / T::GROUP_N) * kargs.stride_sfb + begin + group)[0];
            store<1>(s_sfb, vector_t<u32_t, 1>{raw * 0x01010101u}, group);
        }
    };
    auto read_scale_a = [&](int tile) {
        return static_cast<u32_t>(load<1>(s_sfa,
            (tile & (T::SCALE_PANEL - 1)) * T::SFA_ROWS_PER_REPEAT + wave_m * T::W_M + lane % T::W_M)[0]);
    };
    auto read_scale_b = [&](int tile) {
        return load<1>(s_sfb, tile & (T::SCALE_PANEL - 1))[0];
    };
    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{}, seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
    using MMA = typename decltype(mma)::MMA;
    using AFragment = typename MMA::vtype_a;
    using BFragment = typename MMA::vtype_b;
    array<AFragment, T::E_M> v_a;
    array<BFragment, T::E_N> v_b;
    array<typename MMA::vtype_c, T::E_M * T::E_N> v_c{};

    static_for<T::PREFETCH_DISTANCE>([&](auto slot) {
        if (decltype(slot)::value < loops) prefetch(decltype(slot)::value, decltype(slot)::value);
    });
    publish_scales(0);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    u32_t scale_a = read_scale_a(0), scale_b = read_scale_b(0);
    static_for<T::E_M>([&](auto m) {
        constexpr int i = decltype(m)::value;
        v_a[i] = load_operand_fragment_staged<T::VEC_A, i * T::A_CHUNKS_PER_FRAGMENT, AFragment>(s_a, a_offsets(0));
    });
    static_for<T::E_N>([&](auto n) {
        constexpr int j = decltype(n)::value;
        v_b[j] = load_operand_fragment_staged<T::VEC_B, j * T::B_CHUNKS_PER_FRAGMENT, BFragment>(s_b, layout_to_offsets<T::VEC_B>(u_rb));
    });
    s_waitcnt_lgkmcnt(0_I);
    int stage = 0;
    // The same advance consumes every non-final K group, including the
    // prefetched end of K. Only real future loads are issued, and the wait
    // count follows the number of real queued groups instead of a drain path.
#pragma clang loop unroll_count(T::NUM_STAGES)
    for (int tile = 0; tile + 1 < loops; ++tile) {
        const int next_stage = stage == T::NUM_STAGES - 1 ? 0 : stage + 1;
        const int future_stage = stage == 0 ? T::NUM_STAGES - 1 : stage - 1;
        if (((tile + 1) & (T::SCALE_PANEL - 1)) == 0) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            publish_scales(tile + 1);
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        mma_scale_group<T, 0, 1>(mma, v_a, v_b, v_c, scale_a, scale_b);
        if (tile + T::PREFETCH_DISTANCE < loops)
            prefetch(future_stage, tile + T::PREFETCH_DISTANCE);
        const u32_t next_scale_a = read_scale_a(tile + 1);
        mma_scale_group<T, 1, T::E_N - 1>(mma, v_a, v_b, v_c, scale_a, scale_b);
        if (tile + T::PREFETCH_DISTANCE < loops)
            s_waitcnt_vmcnt(number<T::VMEM_STEADY_WAIT>{});
        else if (T::NUM_STAGES == 4 && tile + 2 < loops)
            s_waitcnt_vmcnt(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
        else
            s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        // Each fragment is replaced immediately after its final current use.
        v_a[0] = load_operand_fragment_staged<T::VEC_A, 0, AFragment>(s_a, a_offsets(next_stage));
        u32_t next_scale_b;
        static_for<T::E_N>([&](auto n) {
            constexpr int j = decltype(n)::value;
            mma_scale_group<T, T::E_N + j, 1>(mma, v_a, v_b, v_c, scale_a, scale_b);
            v_b[j] = load_operand_fragment_staged<T::VEC_B, j * T::B_CHUNKS_PER_FRAGMENT, BFragment>(
                s_b, layout_to_offsets<T::VEC_B>(u_rb + next_stage * T::B_STAGE));
            if constexpr (j == 0) next_scale_b = read_scale_b(tile + 1);
        });
        v_a[1] = load_operand_fragment_staged<T::VEC_A, T::A_CHUNKS_PER_FRAGMENT, AFragment>(s_a, a_offsets(next_stage));
        scale_a = next_scale_a;
        scale_b = next_scale_b;
        stage = next_stage;
    }
    mma_scale_group<T, 0, T::E_M * T::E_N>(mma, v_a, v_b, v_c, scale_a, scale_b);
    const auto coord = opus::make_tuple(wave_m, lane % mma.grpn_c, wave_n, lane / mma.grpn_c);
    const auto output = layout_to_offsets<T::VEC_C>(partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), coord));
    static_for<T::E_M * T::E_N>([&](auto i) {
        constexpr int index = decltype(i)::value;
        store<T::VEC_C>(g_c, cast<bf16_t>(v_c[index]), output[index]);
    });
}
#endif
