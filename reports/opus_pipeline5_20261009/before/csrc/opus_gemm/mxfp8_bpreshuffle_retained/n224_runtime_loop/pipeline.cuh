// SPDX-License-Identifier: Apache-2.0
// Runtime-K 192x224, eight Wave64, native E8M0 and M64 tails.
// Reuses unchanged Apache-2.0 matrix layout helpers; see ../../licenses/.
#pragma once

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "traits.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
__global__ void runtime_n224_loop_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
__global__ __launch_bounds__(runtime_n224_loop_traits::BLOCK_SIZE, runtime_n224_loop_traits::MIN_WGS_PER_CU)
void runtime_n224_loop_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
    using namespace opus;
    using T = runtime_n224_loop_traits;
    const int loops = kargs.k / T::B_K;
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

    // Dispatch guarantees M%64=N%896=0, K%128=0 and K<=1536 and bounded int32 extents.
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

    auto issue_matrix_prefetch = [&](auto stage, auto tile) {
        constexpr int slot = decltype(stage)::value;
        const auto group = tile;
        static_assert(slot >= 0 && slot < T::NUM_STAGES);
        async_load<16>(g_a, s_a.ptr, u_ga, u_sa + slot * T::A_STAGE, group * T::B_K);
        // Fourteen N16 groups: wave w loads w, then w+8 if present.
        // Each N16/K128 group occupies two padded 1024-byte LDS slabs.
        static_for<2>([&](auto pass) {
            const int ng = wave_id + decltype(pass)::value * T::NUM_WAVES;
            if (ng < T::B_N / T::W_N) {
                static_for<2>([&](auto half) {
                    constexpr int h = decltype(half)::value;
                    const int dst = slot * T::B_STAGE + (ng * 2 + h) * (T::smem_linear_wave + T::smem_padding);
                    const int src = ng * T::W_N * kargs.stride_b + h * 1024 + lane_id * 16;
                    async_load<16>(g_b, reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(s_b.ptr + dst)),
                                   src, group * T::B_K * T::W_N);
                });
            }
        });
    };
    auto load_scale_panel = [&]() {
        static_for<T::SFA_PASSES>([&](auto i) {
            const int index = (thread_id_x() + decltype(i)::value * T::BLOCK_SIZE) * 16;
            const int group = index / T::B_M;
            const int local_row = index % T::B_M;
            if (index < T::B_M * loops) {
                vector_t<D_SF, 16> values;
                if (row + local_row < kargs.m)
                    values = load<16>(g_sfa, group * kargs.stride_sfa + row + local_row);
                else
                    static_for<16>([&](auto j) { values[decltype(j)::value] = 0x7f; });
                store<16>(s_sfa, values, index);
            }
        });
        if (thread_id_x() < T::SFB_BYTES) {
            const int half_n = thread_id_x() / T::SCALE_PANEL;
            const int group = thread_id_x() % T::SCALE_PANEL;
            const int global_group = col / T::GROUP_N + half_n;
            const auto value = group < loops && global_group < kargs.n / T::GROUP_N
                ? load<1>(g_sfb, global_group * kargs.stride_sfb + group)[0]
                : static_cast<D_SF>(0x7f);
            store<1>(s_sfb, value, thread_id_x());
        }
    };
    auto read_scales = [&](auto tile, auto& scale_a, auto& scale_b) {
        const auto group = tile;
        static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
        static_for<T::E_M>([&](auto i) {
            constexpr int repeat = decltype(i)::value;
            const int local_row = wave_id_m * T::W_M + lane_id % T::W_M + repeat * T::T_M * T::W_M;
            const unsigned value = load<1>(s_sfa, group * T::B_M + local_row)[0];
            scale_a[repeat / 4] |= value << ((repeat % 4) * 8);
        });
        static_for<T::B_SCALE_PACKS>([&](auto i) {
            constexpr int ni = decltype(i)::value;
            const int scale_group = ((col % T::GROUP_N) + (ni * T::T_N + wave_id_n) * T::W_N) / T::GROUP_N;
            scale_b[ni] = load<1>(s_sfb, scale_group * T::SCALE_PANEL + group)[0];
        });
    };
    auto load_a = [&](auto m, auto stage) {
        constexpr int repeat = decltype(m)::value;
        constexpr int slot = decltype(stage)::value;
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            const int offset = T::a_lds_offset(wave_id_m, lane_id, repeat, decltype(c)::value) + slot * T::A_STAGE;
            a_chunks[index] = load<16>(s_a, offset);
        });
    };
    auto load_b = [&](auto n, auto stage) {
        constexpr int repeat = decltype(n)::value;
        constexpr int slot = decltype(stage)::value;
        const auto offsets = layout_to_offsets<16>(u_rb + slot * T::B_STAGE);
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            b_chunks[index] = load<16>(s_b, offsets[index]);
        });
    };
    auto compute = [&](auto m, auto n) {
        constexpr int mi = decltype(m)::value, ni = decltype(n)::value;
        constexpr int ci = mi * T::E_N + ni;
        constexpr int half_n = ni;
        c00[ci] = BaseMMA{}(v_a[mi], v_b[ni], c00[ci],
                           static_cast<int>(v_sfa[mi / 4]), static_cast<int>(v_sfb[half_n]),
                           number<mi % 4>{}, number<0>{});
    };

    // Seed K0/K1 once. The complete scale panel has no runtime refill path.
    load_scale_panel();
    issue_matrix_prefetch(number<0>{}, number<0>{});
    if (loops > 1) issue_matrix_prefetch(number<1>{}, number<1>{});
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(number<0>{}, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m) { load_a(m, number<0>{}); });
    static_for<T::E_N>([&](auto n) { load_b(n, number<0>{}); });
    s_waitcnt_lgkmcnt(0_I);
    // Every wave must finish reading K0 before any producer reuses its slot.
    __builtin_amdgcn_s_barrier();

    auto advance = [&](auto stage, auto next_stage, int group) {
        // Current operands are in registers; prefetch only an existing K+2.
        if (group + 2 < loops)
            issue_matrix_prefetch(stage, group + 2);
        read_scales(group + 1, v_sfa_next, v_sfb_next);
        static_for<T::E_M>([&](auto m) {
            static_for<T::E_N>([&](auto n) {
                compute(m, n);
                if constexpr (decltype(m)::value == T::E_M - 1)
                    load_b(n, next_stage);
            });
            load_a(m, next_stage);
        });
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    };
    // One runtime loop contains both static LDS stages and the drain.
    // K128 skips the loop; even group counts finish after stage0, odd after
    // stage1. The last group's register operands feed the fused final MFMA.
#pragma clang loop unroll(disable)
    for (int group = 0; group + 1 < loops; group += 2) {
        advance(number<0>{}, number<1>{}, group);
        if (group + 2 < loops)
            advance(number<1>{}, number<0>{}, group + 1);
    }

    // K128's operand-read barrier, or the final advance, has finished every
    // matrix LDS reader. Final MFMA consumes registers before BF16 staging.
    constexpr int C_PITCH = T::B_N + 8;
    constexpr int C_BYTES = T::B_M * C_PITCH * sizeof(D_C);
    constexpr int OUTPUT_VEC = 4;
    constexpr int OUTPUT_PASSES = T::B_M * T::B_N / (T::BLOCK_SIZE * OUTPUT_VEC);
    static_assert(C_PITCH == 232 && C_BYTES == 89088 && C_BYTES <= T::LDS_BYTES);
    static_assert(T::B_M * T::B_N % (T::BLOCK_SIZE * OUTPUT_VEC) == 0);
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem));
    const auto p_coord = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(C_PITCH, 1_I), p_coord);
    const auto c_offsets = layout_to_offsets<T::VEC_C>(u_sc);
    // Final-group operands are already in registers on all waves. Stage
    // each completed accumulator while independent final MFMAs continue.
    static_for<T::E_M>([&](auto m) {
        static_for<T::E_N>([&](auto n) {
            constexpr int index = decltype(m)::value * T::E_N + decltype(n)::value;
            compute(m, n);
            store<T::VEC_C>(s_c, cast<D_C>(c00[index]), c_offsets[index]);
        });
    });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    // Recombine MFMA's scattered rows into contiguous 16-byte global stores.
    static_for<OUTPUT_PASSES>([&](auto pass) {
        const int linear = thread_id_x() * OUTPUT_VEC + decltype(pass)::value * T::BLOCK_SIZE * OUTPUT_VEC;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        const auto value = load<OUTPUT_VEC>(s_c, output_row * C_PITCH + output_col);
        store<OUTPUT_VEC>(g_c, value, output_row * kargs.stride_c + output_col, 0, opus::number<0>{});
    });

}
#endif
