// SPDX-License-Identifier: Apache-2.0
// Runtime-K 64x128 flow on four Wave64s: one priming pass,
// one advancing loop and one final compute/output.
#pragma once
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "traits_runtime.cuh"

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
    using ScaleWords = vector_t<D_SF_PACK, 4>;

    // Tile and thread coordinates.
    // Keep wave-uniform values before tile traversal, preserving their lowering.
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M, wave_id_n = wave_id / T::T_M;
    const unsigned grid_m = kargs.m / T::B_M, grid_n = kargs.n / T::B_N;
    const unsigned total = grid_m * grid_n;
    unsigned block_m = block_id_y(), block_n = block_id_x();
    const unsigned linear = block_m * grid_n + block_n;
    // Geometry-only traversal retains the measured general partition rule
    // for 128-column tiles. The policy does not depend on K or an enumerated shape.
    if (T::B_N == 64 && grid_n <= 16u && (grid_n & 3u) == 0u && grid_m >= 16u && grid_m <= 32u) {
        const unsigned n_per_partition = grid_n / 4;
        const unsigned full_pairs = (grid_m & ~1u) * grid_n;
        if (linear < full_pairs) {
            const unsigned partition = linear & 7u, local = linear >> 3;
            block_m = (local / n_per_partition) * 2 + partition / 4;
            block_n = (partition % 4) * n_per_partition + local % n_per_partition;
        } else {
            block_m = grid_m - 1;
            block_n = linear - full_pairs;
        }
    } else if (grid_n <= 16u && total > 256u) {
        const unsigned partitions = ((grid_n <= 8u && total < 768u) || grid_m <= 24u) ? 16u : 4u;
        if (total % partitions == 0u) {
            const unsigned logical = (linear % partitions) * (total / partitions) + linear / partitions;
            block_m = logical / grid_n;
            block_n = logical % grid_n;
        }
    }
    const int row = block_m * T::B_M, col = block_n * T::B_N;

    const int loops = kargs.k / T::B_K;

    const int batch_id = block_id_z();

    // Matrix global-memory views.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + batch_id * kargs.stride_a_batch + row * kargs.stride_a);
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + batch_id * kargs.stride_b_batch + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + batch_id * kargs.stride_c_batch + row * kargs.stride_c + col);
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa) + batch_id * kargs.stride_sfa_batch + row);
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb) + batch_id * kargs.stride_sfb_batch + (col / T::GROUP_N) * kargs.stride_sfb);

    // Matrix layouts: global -> LDS -> registers.
    const int producer_row = lane_id / (T::B_K / T::VEC_A);
    const int producer_xor = (producer_row & 1) | ((producer_row & 2) << 1);
    const auto u_ga = make_layout_ga_scale<T>(lane_id ^ producer_xor, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);

    // Matrix and scale LDS.
    alignas(16) __shared__ char smem_matrix[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem_matrix));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem_matrix + T::NUM_STAGES * T::A_STAGE));
    auto s_sfa = make_smem(reinterpret_cast<u16_t*>(smem_matrix + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF_PACK*>(smem_matrix + T::MATRIX_LDS_BYTES + T::SFA_BYTES));

    // MMA and register fragments.
    auto mma = make_tiled_mma<D_A, D_B, D_ACC>(seq<T::E_M, T::E_N, T::E_K>{},
        seq<T::T_M, T::T_N, T::T_K>{}, seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
    using BaseMMA = typename decltype(mma)::MMA;
    using AFragment = typename BaseMMA::vtype_a;
    using BFragment = typename BaseMMA::vtype_b;
    using AccFragment = typename BaseMMA::vtype_c;
    array<AFragment, T::E_M> v_a;
    array<BFragment, T::E_N> v_b;
    array<AccFragment, T::E_M * T::E_N> c00{};

    D_SF_PACK v_sfa, v_sfb;

    // Matrix offsets for global K tiles and LDS ring stages.
    auto ga_offset = [&](int tile) { return tile * T::B_K; };
    auto gb_offset = [&](int tile) { return tile * T::B_K * T::W_N; };
    auto sa_offset = [&](int stage) { return stage * T::A_STAGE; };
    auto sb_offset = [&](int stage) { return stage * T::B_STAGE; };

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
    // A/B K-tile requests copy global memory directly into an LDS ring slot.
    auto issue_matrix_prefetch = [&](int stage, int tile) {
        async_load<T::VEC_A>(g_a, s_a.ptr, u_ga, u_sa + sa_offset(stage), ga_offset(tile));
        async_load<T::VEC_B>(g_b, s_b.ptr, u_gb, u_sb + sb_offset(stage), gb_offset(tile));
    };

    // Scale-panel publication and packed register reads.
    auto load_scale_panel = [&](int begin) {
        const int group = thread_id_x() / T::SFA_PRODUCERS_PER_GROUP;
        const int pair_row = (thread_id_x() % T::SFA_PRODUCERS_PER_GROUP) * T::VEC_SCALE_A;
        if (group < T::SCALE_PANEL && begin + group < loops) {
            const int offset = (begin + group) * kargs.stride_sfa + pair_row;
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
    // Replace an LDS operand fragment immediately after its last current use.
    auto load_a = [&](auto m, int stage) {
        constexpr int i = decltype(m)::value;
        v_a[i] = load_operand_fragment_staged<T::VEC_A, i * T::A_CHUNKS_PER_FRAGMENT, AFragment>(s_a, a_offsets(stage));
    };
    auto load_b = [&](auto n, int stage) {
        constexpr int j = decltype(n)::value;
        v_b[j] = load_operand_fragment_staged<T::VEC_B, j * T::B_CHUNKS_PER_FRAGMENT, BFragment>(
            s_b, layout_to_offsets<T::VEC_B>(u_rb + sb_offset(stage)));
    };

    int stage = 0;

    // ===== Prologue =====
    static_for<T::PREFETCH_DISTANCE>([&](auto slot) {
        if (decltype(slot)::value < loops) issue_matrix_prefetch(decltype(slot)::value, decltype(slot)::value);
    });
    load_scale_panel(0);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    v_sfa = read_scale_a(0);
    v_sfb = read_scale_b(0);
    static_for<T::E_M>([&](auto m) { load_a(m, 0); });
    static_for<T::E_N>([&](auto n) { load_b(n, 0); });
    s_waitcnt_lgkmcnt(0_I);

    // ===== Main loop =====
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
            load_scale_panel(tile + 1);
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        mma_scale_group<T, 0, 1>(mma, v_a, v_b, c00, v_sfa, v_sfb);
        if (tile + T::PREFETCH_DISTANCE < loops)
            issue_matrix_prefetch(future_stage, tile + T::PREFETCH_DISTANCE);
        const D_SF_PACK v_sfa_next = read_scale_a(tile + 1);
        mma_scale_group<T, 1, T::E_N - 1>(mma, v_a, v_b, c00, v_sfa, v_sfb);
        if (tile + T::PREFETCH_DISTANCE < loops)
            s_waitcnt_vmcnt(number<T::VMEM_STEADY_WAIT>{});
        else if (T::NUM_STAGES == 4 && tile + 2 < loops)
            s_waitcnt_vmcnt(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
        else
            s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        // Each fragment is replaced immediately after its final current use.
        load_a(number<0>{}, next_stage);
        D_SF_PACK v_sfb_next;
        static_for<T::E_N>([&](auto n) {
            constexpr int j = decltype(n)::value;
            mma_scale_group<T, T::E_N + j, 1>(mma, v_a, v_b, c00, v_sfa, v_sfb);
            load_b(n, next_stage);
            if constexpr (j == 0) v_sfb_next = read_scale_b(tile + 1);
        });
        load_a(number<1>{}, next_stage);
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
        stage = next_stage;
    }

    // ===== Epilogue =====
    mma_scale_group<T, 0, T::E_M * T::E_N>(mma, v_a, v_b, c00, v_sfa, v_sfb);

    // ===== Output writeback =====
    const auto p_coord_c = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord_c);
    const auto gc_offsets = layout_to_offsets<T::VEC_C>(u_gc);
    static_for<T::E_M * T::E_N>([&](auto i) {
        constexpr int index = decltype(i)::value;
        store<T::VEC_C>(g_c, cast<D_C>(c00[index]), gc_offsets[index]);
    });
}
#endif
