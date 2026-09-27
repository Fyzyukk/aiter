// SPDX-License-Identifier: Apache-2.0
// Runtime K128..16384, 128/160x128, four Wave64, native E8M0 and M64 tails.
// Reuses unchanged Apache-2.0 matrix layout helpers from csrc/opus_gemm/include/gfx950.
#pragma once

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "traits.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void small_flow_v4_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void small_flow_v4_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;
    using T = remove_cvref_t<Traits>;
    using D_A = fp8_t;
    using D_B = fp8_t;
    using D_C = bf16_t;
    using D_SF = unsigned char;
    const int wave_id = __builtin_amdgcn_readfirstlane(thread_id_x() / T::WARP_SIZE);
    const int lane_id = thread_id_x() % T::WARP_SIZE;
    const int wave_id_m = wave_id % T::T_M;
    const int wave_id_n = wave_id / T::T_M;
    const int row = block_id_y() * T::B_M;
    const int col = block_id_x() * T::B_N;
    const int k_tiles = kargs.k / T::B_K;

    // Dispatch guarantees M%64=N%128=K%128=0, 128<=K<=16384 and int32 extents.
    // Buffer OOB zero-fills missing A rows and discards missing output rows.
    auto g_a = make_gmem(reinterpret_cast<const D_A*>(kargs.ptr_a) + row * kargs.stride_a,
                        static_cast<unsigned>((kargs.m - row) * kargs.stride_a));
    auto g_b = make_gmem(reinterpret_cast<const D_B*>(kargs.ptr_b) + col * kargs.stride_b);
    auto g_c = make_gmem(reinterpret_cast<D_C*>(kargs.ptr_c) + row * kargs.stride_c + col,
                        static_cast<unsigned>(((kargs.m - row) * kargs.stride_c - col) * sizeof(D_C)));
    auto g_sfa = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfa));
    auto g_sfb = make_gmem(reinterpret_cast<const D_SF*>(kargs.ptr_sfb));
    alignas(16) __shared__ char smem[T::LDS_BYTES];
    auto s_a = make_smem(reinterpret_cast<D_A*>(smem));
    auto s_b = make_smem(reinterpret_cast<D_B*>(smem + T::NUM_STAGES * T::A_STAGE));
    auto s_sfa = make_smem(reinterpret_cast<D_SF*>(smem + T::MATRIX_LDS_BYTES));
    auto s_sfb = make_smem(reinterpret_cast<D_SF*>(smem + T::MATRIX_LDS_BYTES + T::SFA_BYTES));
    const auto u_ga = make_layout_ga_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_a);
    const auto u_sa = make_layout_sa_scale<T>(wave_id_m, wave_id_n);
    const auto u_gb = make_layout_gb_scale<T>(lane_id, wave_id_m, wave_id_n, kargs.stride_b);
    const auto u_sb = make_layout_sb_scale<T>(wave_id_m, wave_id_n);
    const auto u_rb = make_layout_rb_scale<T>(lane_id, wave_id_n);
    auto mma = make_tiled_mma<D_A, D_B, fp32_t>(
        seq<T::E_M, T::E_N, 1>{}, seq<T::T_M, T::T_N, 1>{},
        seq<16, 16, 128>{}, mfma_adaptor_swap_ab{});
    using BaseMMA = typename decltype(mma)::MMA;
    using AFragment = typename BaseMMA::vtype_a;
    using BFragment = typename BaseMMA::vtype_b;
    using AccFragment = typename BaseMMA::vtype_c;
    using Chunk = vector_t<fp8_t, 16>;
    array<AFragment, T::E_M> v_a;
    array<BFragment, T::E_N> v_b;
    array<AccFragment, T::E_M * T::E_N> c00{};
    array<unsigned, T::A_SCALE_PACKS> v_sfa{}, v_sfa_next{};
    array<unsigned, T::B_SCALE_PACKS> v_sfb{}, v_sfb_next{};
    auto* a_chunks = reinterpret_cast<Chunk*>(&v_a);
    auto* b_chunks = reinterpret_cast<Chunk*>(&v_b);

    auto issue_matrix_prefetch = [&](int slot, int group) {
        async_load<16>(g_a, s_a.ptr, u_ga, u_sa + slot * T::A_STAGE, group * T::B_K);
        async_load<16>(g_b, s_b.ptr, u_gb, u_sb + slot * T::B_STAGE, group * T::B_K * T::W_N);
    };
    auto load_scale_panel = [&](int panel_begin) {
        static_for<T::SFA_PASSES>([&](auto i) {
            const int index = (thread_id_x() + decltype(i)::value * T::BLOCK_SIZE) * 16;
            const int group = index / T::B_M;
            const int local_row = index % T::B_M;
            if (index < T::SFA_BYTES && panel_begin + group < k_tiles) {
                vector_t<D_SF, 16> values;
                if (row + local_row < kargs.m)
                    values = load<16>(g_sfa, (panel_begin + group) * kargs.stride_sfa + row + local_row);
                else
                    static_for<16>([&](auto j) { values[decltype(j)::value] = 0x7f; });
                store<16>(s_sfa, values, index);
            }
        });
        // BN128 has exactly one B-scale row. Do not load unused K groups.
        if (thread_id_x() < T::SCALE_PANEL && panel_begin + thread_id_x() < k_tiles) {
            const int group = thread_id_x();
            store<1>(s_sfb, load<1>(g_sfb, (col / T::GROUP_N) * kargs.stride_sfb + panel_begin + group),
                     thread_id_x());
        }
    };
    auto read_scales = [&](int group, auto& scale_a, auto& scale_b) {
        const int panel_group = group & (T::SCALE_PANEL - 1);
        static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
        static_for<T::E_M>([&](auto i) {
            constexpr int repeat = decltype(i)::value;
            const int local_row = wave_id_m * T::W_M + lane_id % T::W_M + repeat * T::T_M * T::W_M;
            const unsigned value = load<1>(s_sfa, panel_group * T::B_M + local_row)[0];
            scale_a[repeat / 4] |= value << ((repeat % 4) * 8);
        });
        static_for<T::B_SCALE_PACKS>([&](auto i) {
            constexpr int half_n = decltype(i)::value;
            scale_b[half_n] = load<1>(s_sfb, half_n * T::SCALE_PANEL + panel_group)[0];
        });
    };
    auto load_a = [&](auto m, int slot) {
        constexpr int repeat = decltype(m)::value;
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            const int offset = T::a_lds_offset(wave_id_m, lane_id, repeat, decltype(c)::value) + slot * T::A_STAGE;
            // BM128: C AGPR0:63, A64:95. BM160: C AGPR0:79, A80:119.
            [[clang::amdgpu_pin_agpr(T::C_REGS + index * 4)]]
            a_chunks[index] = load<16>(s_a, offset);
        });
    };
    auto load_b = [&](auto n, int slot) {
        constexpr int repeat = decltype(n)::value;
        const auto offsets = layout_to_offsets<16>(u_rb + slot * T::B_STAGE);
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            b_chunks[index] = load<16>(s_b, offsets[index]);
        });
    };
    auto compute = [&](auto m, auto n) {
        constexpr int mi = decltype(m)::value, ni = decltype(n)::value;
        constexpr int ci = mi * T::E_N + ni;
        constexpr int half_n = ni / (T::GROUP_N / (T::T_N * T::W_N));
        [[clang::amdgpu_pin_agpr(ci * sizeof(AccFragment) / sizeof(u32_t))]]
        c00[ci] = BaseMMA{}(v_a[mi], v_b[ni], c00[ci],
                           static_cast<int>(v_sfa[mi / 4]), static_cast<int>(v_sfb[half_n]),
                           number<mi % 4>{}, number<0>{});
    };

    // Seed only existing groups. One-tile K128 never loads or reads K1.
    load_scale_panel(0);
    issue_matrix_prefetch(0, 0);
    if (k_tiles > 1)
        issue_matrix_prefetch(1, 1);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(0, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m) { load_a(m, 0); });
    static_for<T::E_N>([&](auto n) { load_b(n, 0); });
    s_waitcnt_lgkmcnt(0_I);
    // S2 reuses K0 immediately for K2. S3 first writes the independent K2
    // slot; the first publication barrier retires all K0 readers before K3.
    if constexpr (T::NUM_STAGES == 2)
        __builtin_amdgcn_s_barrier();

    auto advance = [&](int stage, int group) {
        const int next_stage = stage + 1 == T::NUM_STAGES ? 0 : stage + 1;
        const int future_stage = T::NUM_STAGES == 2 ? stage : (stage == 0 ? 2 : stage - 1);
        // All tile shapes prefetch K+2 into its ring slot. The stage count is
        // a tile-geometry constant; runtime K never selects a different flow.
        const bool has_future = group + 2 < k_tiles;
        if (has_future)
            issue_matrix_prefetch(future_stage, group + 2);
        // Current scales were read before the preceding publication barrier,
        // so all waves have retired those panel reads even with the S3 ring.
        // The current group's scales are in registers while the panel refills.
        if (((group + 1) & (T::SCALE_PANEL - 1)) == 0) {
            load_scale_panel(group + 1);
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        read_scales(group + 1, v_sfa_next, v_sfb_next);
        // Preserve the original last-use replacement of each operand.
        static_for<T::E_M>([&](auto m) {
            static_for<T::E_N>([&](auto n) {
                compute(m, n);
                if constexpr (decltype(m)::value == T::E_M - 1)
                    load_b(n, next_stage);
            });
            if constexpr (decltype(m)::value == 0) {
                // The first M repeat consumes only current register operands.
                // Before the first next-tile LDS read, retire the older K+1
                // matrix requests while permitting this step's K+2 requests
                // to stay in flight. The no-future drain must wait for all.
                __builtin_amdgcn_sched_barrier(0);
                if (has_future)
                    s_waitcnt_vmcnt(number<T::VMEM_INSTRUCTIONS_PER_TILE>{});
                else
                    s_waitcnt_vmcnt(0_I);
                s_waitcnt_lgkmcnt(0_I);
                __builtin_amdgcn_s_barrier();
                __builtin_amdgcn_sched_barrier(0);
            }
            load_a(m, next_stage);
        });
        // S2's next advance immediately reuses the just-read next slot for
        // K+3 and needs a consumer barrier. S3 instead writes K+3 into the
        // older current slot: this advance's publication barrier already
        // retired every reader of that older slot, so no end barrier is needed.
        if constexpr (T::NUM_STAGES == 2) {
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    };

    // One runtime U1 body owns the matrix ring and drain. Track the scalar
    // stage with one increment/wrap; no per-stage or fixed-K loop copies.
    int stage = 0;
#pragma clang loop unroll(disable)
    for (int group = 0; group + 1 < k_tiles; ++group) {
        advance(stage, group);
        stage = stage + 1 == T::NUM_STAGES ? 0 : stage + 1;
    }
    static_for<T::E_M>([&](auto m) {
        static_for<T::E_N>([&](auto n) { compute(m, n); });
    });

    // The seed or final no-future advance drained all matrix VMEM requests.
    // S2 already retired the last operand readers at its final end barrier.
    // S3 defers that one consumer barrier to here, before LDS becomes output.
    if constexpr (T::NUM_STAGES == 3) {
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
    }
    constexpr int C_PITCH = T::B_N + 8;
    constexpr int C_BYTES = T::B_M * C_PITCH * sizeof(D_C);
    constexpr int OUTPUT_VEC = 8;
    constexpr int OUTPUT_PASSES = T::B_M * T::B_N / (T::BLOCK_SIZE * OUTPUT_VEC);
    static_assert(C_PITCH == 136 && C_BYTES <= T::LDS_BYTES);
    static_assert(T::B_M * T::B_N % (T::BLOCK_SIZE * OUTPUT_VEC) == 0);
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem));
    const auto p_coord = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(C_PITCH, 1_I), p_coord);
    const auto c_offsets = layout_to_offsets<T::VEC_C>(u_sc);
    static_for<T::E_M * T::E_N>([&](auto i) {
        constexpr int index = decltype(i)::value;
        store<T::VEC_C>(s_c, cast<D_C>(c00[index]), c_offsets[index]);
    });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    // Recombine MFMA's scattered rows into contiguous 16-byte global stores.
    static_for<OUTPUT_PASSES>([&](auto pass) {
        const int linear = thread_id_x() * OUTPUT_VEC + decltype(pass)::value * T::BLOCK_SIZE * OUTPUT_VEC;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        const auto value = load<OUTPUT_VEC>(s_c, output_row * C_PITCH + output_col);
        store<OUTPUT_VEC>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<2>{});
    });

}
#endif
