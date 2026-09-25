// SPDX-License-Identifier: Apache-2.0
// Fixed-K 192x256, eight Wave64, native E8M0 and M64 tails.
// Reuses unchanged Apache-2.0 matrix layout helpers; see ../../licenses/.
#pragma once

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_bpreshuffle_shortk_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void gemm_a8w8_mxfp8_bpreshuffle_shortk_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
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

    // Dispatch guarantees M%64=N%256=0, K==FIXED_K and bounded int32 extents.
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

    auto issue_matrix_prefetch = [&](auto stage, auto tile) {
        constexpr int slot = decltype(stage)::value;
        constexpr int group = decltype(tile)::value;
        static_assert(slot >= 0 && slot < T::NUM_STAGES);
        static_assert(group >= 0 && group < T::K_TILES);
        async_load<16>(g_a, s_a.ptr, u_ga, u_sa + slot * T::A_STAGE, group * T::B_K);
        async_load<16>(g_b, s_b.ptr, u_gb, u_sb + slot * T::B_STAGE, group * T::B_K * T::W_N);
    };
    auto load_scale_panel = [&]() {
        static_for<T::SFA_PASSES>([&](auto i) {
            const int index = (thread_id_x() + decltype(i)::value * T::BLOCK_SIZE) * 16;
            const int group = index / T::B_M;
            const int local_row = index % T::B_M;
            if (index < T::SFA_BYTES) {
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
            store<1>(s_sfb, load<1>(g_sfb, (col / T::GROUP_N + half_n) * kargs.stride_sfb + group),
                     thread_id_x());
        }
    };
    auto read_scales = [&](auto tile, auto& scale_a, auto& scale_b) {
        constexpr int group = decltype(tile)::value;
        static_assert(group >= 0 && group < T::K_TILES);
        static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
        static_for<T::E_M>([&](auto i) {
            constexpr int repeat = decltype(i)::value;
            const int local_row = wave_id_m * T::W_M + lane_id % T::W_M + repeat * T::T_M * T::W_M;
            const unsigned value = load<1>(s_sfa, group * T::B_M + local_row)[0];
            scale_a[repeat / 4] |= value << ((repeat % 4) * 8);
        });
        static_for<T::B_SCALE_PACKS>([&](auto i) {
            constexpr int half_n = decltype(i)::value;
            scale_b[half_n] = load<1>(s_sfb, half_n * T::SCALE_PANEL + group)[0];
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
        constexpr int half_n = ni / (T::GROUP_N / (T::T_N * T::W_N));
        c00[ci] = BaseMMA{}(v_a[mi], v_b[ni], c00[ci],
                           static_cast<int>(v_sfa[mi / 4]), static_cast<int>(v_sfb[half_n]),
                           number<mi % 4>{}, number<0>{});
    };

    // Seed K0/K1 once. The complete scale panel has no runtime refill path.
    load_scale_panel();
    issue_matrix_prefetch(number<0>{}, number<0>{});
    issue_matrix_prefetch(number<1>{}, number<1>{});
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(number<0>{}, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m) { load_a(m, number<0>{}); });
    static_for<T::E_N>([&](auto n) { load_b(n, number<0>{}); });
    s_waitcnt_lgkmcnt(0_I);
    // Every wave must finish reading K0 before any producer reuses its slot.
    __builtin_amdgcn_s_barrier();

    static_for<T::K_TILES - 1>([&](auto tile) {
        constexpr int group = decltype(tile)::value;
        constexpr int stage = group % T::NUM_STAGES;
        constexpr int next_stage = (group + 1) % T::NUM_STAGES;
        // Current operands are in registers; K+2 can replace their LDS slot.
        // Do not instantiate a prefetch beyond the final fixed-K group.
        if constexpr (group + 2 < T::K_TILES)
            issue_matrix_prefetch(number<stage>{}, number<group + 2>{});
        read_scales(number<group + 1>{}, v_sfa_next, v_sfb_next);
        // Replace each A after its last N use, and B after the last M use.
        static_for<T::E_M>([&](auto m) {
            static_for<T::E_N>([&](auto n) {
                compute(m, n);
                if constexpr (decltype(m)::value == T::E_M - 1)
                    load_b(n, number<next_stage>{});
            });
            load_a(m, number<next_stage>{});
        });
        // Publish the future slot and finish all reads before slot reuse.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
    });
    static_for<T::E_M>([&](auto m) {
        static_for<T::E_N>([&](auto n) { compute(m, n); });
    });

    const auto p_coord = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_gc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(kargs.stride_c, 1_I), p_coord);
    const auto offsets = layout_to_offsets<T::VEC_C>(u_gc);
    static_for<T::E_M * T::E_N>([&](auto i) {
        constexpr int index = decltype(i)::value;
        store<T::VEC_C>(g_c, cast<D_C>(c00[index]), offsets[index]);
    });
}
#endif
