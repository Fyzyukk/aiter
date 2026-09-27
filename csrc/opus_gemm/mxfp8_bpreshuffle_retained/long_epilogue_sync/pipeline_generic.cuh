// SPDX-License-Identifier: Apache-2.0
// Experimental 192x256, four/eight Wave64, native E8M0 and M64 tails.
// Reuses the imported Apache-2.0 matrix layout helpers; see ../../licenses/.
#pragma once

#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_192x256_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class Traits>
__global__ void gemm_a8w8_mxfp8_bpreshuffle_generic_no_repeat_sync_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::MIN_WGS_PER_CU)
void gemm_a8w8_mxfp8_bpreshuffle_generic_no_repeat_sync_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 kargs) {
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
    const int loops = kargs.k / T::B_K;

    // The launch contract is M%64=N%256=K%128=0, with bounded int32 extents.
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

    auto issue_matrix_prefetch = [&](int stage, int tile) {
        async_load<16>(g_a, s_a.ptr, u_ga, u_sa + stage * T::A_STAGE, tile * T::B_K);
        async_load<16>(g_b, s_b.ptr, u_gb, u_sb + stage * T::B_STAGE, tile * T::B_K * T::W_N);
    };
    auto load_scale_panel = [&](int begin) {
        static_for<T::SFA_PASSES>([&](auto i) {
            const int index = (thread_id_x() + decltype(i)::value * T::BLOCK_SIZE) * 16;
            const int group = index / T::B_M;
            const int local_row = index % T::B_M;
            if (index < T::SFA_BYTES && begin + group < loops) {
                vector_t<D_SF, 16> values;
                if (row + local_row < kargs.m)
                    values = load<16>(g_sfa, (begin + group) * kargs.stride_sfa + row + local_row);
                else
                    static_for<16>([&](auto j) { values[decltype(j)::value] = 0x7f; });
                store<16>(s_sfa, values, index);
            }
        });
        if (thread_id_x() < T::SFB_BYTES) {
            const int half_n = thread_id_x() / T::SCALE_PANEL;
            const int group = thread_id_x() % T::SCALE_PANEL;
            if (begin + group < loops)
                store<1>(s_sfb, load<1>(g_sfb, (col / T::GROUP_N + half_n) * kargs.stride_sfb + begin + group),
                         thread_id_x());
        }
    };
    auto read_scales = [&](int tile, auto& scale_a, auto& scale_b) {
        static_for<T::A_SCALE_PACKS>([&](auto p) { scale_a[decltype(p)::value] = 0; });
        static_for<T::E_M>([&](auto i) {
            constexpr int repeat = decltype(i)::value;
            const int local_row = wave_id_m * T::W_M + lane_id % T::W_M + repeat * T::T_M * T::W_M;
            const unsigned value = load<1>(s_sfa, (tile % T::SCALE_PANEL) * T::B_M + local_row)[0];
            scale_a[repeat / 4] |= value << ((repeat % 4) * 8);
        });
        static_for<T::B_SCALE_PACKS>([&](auto i) {
            constexpr int half_n = decltype(i)::value;
            scale_b[half_n] = load<1>(s_sfb, half_n * T::SCALE_PANEL + tile % T::SCALE_PANEL)[0];
        });
    };
    auto load_a = [&](auto m, int stage) {
        constexpr int repeat = decltype(m)::value;
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            const int offset = T::a_lds_offset(wave_id_m, lane_id, repeat, decltype(c)::value) + stage * T::A_STAGE;
            if constexpr (T::NUM_WAVES == 4) {
                [[clang::amdgpu_pin_agpr(T::C_REGS + index * 4)]]
                a_chunks[index] = load<16>(s_a, offset);
            } else {
                a_chunks[index] = load<16>(s_a, offset);
            }
        });
    };
    auto load_b = [&](auto n, int stage) {
        constexpr int repeat = decltype(n)::value;
        const auto offsets = layout_to_offsets<16>(u_rb + stage * T::B_STAGE);
        static_for<2>([&](auto c) {
            constexpr int index = repeat * 2 + decltype(c)::value;
            b_chunks[index] = load<16>(s_b, offsets[index]);
        });
    };
    auto compute = [&](auto m, auto n) {
        constexpr int mi = decltype(m)::value, ni = decltype(n)::value;
        constexpr int ci = mi * T::E_N + ni;
        constexpr int half_n = ni / (T::GROUP_N / (T::T_N * T::W_N));
        if constexpr (T::NUM_WAVES == 4) {
            [[clang::amdgpu_pin_agpr(ci * sizeof(AccFragment) / sizeof(u32_t))]]
            c00[ci] = BaseMMA{}(v_a[mi], v_b[ni], c00[ci],
                               static_cast<int>(v_sfa[mi / 4]), static_cast<int>(v_sfb[half_n]),
                               number<mi % 4>{}, number<0>{});
        } else {
            c00[ci] = BaseMMA{}(v_a[mi], v_b[ni], c00[ci],
                               static_cast<int>(v_sfa[mi / 4]), static_cast<int>(v_sfb[half_n]),
                               number<mi % 4>{}, number<0>{});
        }
    };

    // K0 is in registers, K1 is ready in LDS. At each iteration K+2 replaces
    // the consumed LDS slot while current operands are consumed by MFMA.
    load_scale_panel(0);
    issue_matrix_prefetch(0, 0);
    if (loops > 1) issue_matrix_prefetch(1, 1);
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    read_scales(0, v_sfa, v_sfb);
    static_for<T::E_M>([&](auto m) { load_a(m, 0); });
    static_for<T::E_N>([&](auto n) { load_b(n, 0); });
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    int stage = 0, tile = 0;
    auto step = [&](auto prefetch) {
        const int next_stage = stage ^ 1;
        if ((tile + 1) % T::SCALE_PANEL == 0) {
            // Current scales are already in registers on every wave.
            load_scale_panel(tile + 1);
            s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        if constexpr (decltype(prefetch)::value) issue_matrix_prefetch(stage, tile + 2);
        read_scales(tile + 1, v_sfa_next, v_sfb_next);
        // Replace each A after its last N use; replace B after the last M use.
        static_for<T::E_M>([&](auto m) {
            static_for<T::E_N>([&](auto n) {
                compute(m, n);
                if constexpr (decltype(m)::value == T::E_M - 1) load_b(n, next_stage);
            });
            load_a(m, next_stage);
        });
        // All waves finish reading the next slot before it can be reused;
        // all producer waves publish the future slot before its consumers.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        v_sfa = v_sfa_next;
        v_sfb = v_sfb_next;
        stage = next_stage;
    };
#pragma unroll 1
    for (; tile + 2 < loops; ++tile) step(true_type{});
    if (tile + 1 < loops) { step(false_type{}); ++tile; }
    static_for<T::E_M>([&](auto m) {
        static_for<T::E_N>([&](auto n) { compute(m, n); });
    });
    // Final compute uses registers only. For K>=256, the final step(false)
    // already waits/barriers after all final A/B/scale LDS reads. For K128,
    // the prologue's second wait/barrier provides the same guarantee.
    // No matrix or scale LDS access occurs after those barriers.
    constexpr int C_PITCH = T::B_N + 8;
    constexpr int OUTPUT_PASSES = T::B_M * T::B_N / (T::BLOCK_SIZE * 8);
    static_assert(T::B_M * C_PITCH * sizeof(D_C) <= T::LDS_BYTES);
    static_assert(T::B_M * T::B_N % (T::BLOCK_SIZE * 8) == 0);
    auto s_c = make_smem(reinterpret_cast<D_C*>(smem));
    const auto p_coord = opus::make_tuple(wave_id_m, lane_id % mma.grpn_c, wave_id_n, lane_id / mma.grpn_c);
    const auto u_sc = partition_layout_c<T::VEC_C>(mma, opus::make_tuple(number<C_PITCH>{}, 1_I), p_coord);
    const auto offsets = layout_to_offsets<T::VEC_C>(u_sc);
    static_for<T::E_M * T::E_N>([&](auto i) {
        constexpr int index = decltype(i)::value;
        store<T::VEC_C>(s_c, cast<D_C>(c00[index]), offsets[index]);
    });
    // Publish the complete BF16 tile before cooperative contiguous reads.
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    const int output_thread = wave_id * T::WARP_SIZE + lane_id;
    static_for<OUTPUT_PASSES>([&](auto i) {
        constexpr int pass = decltype(i)::value;
        const int linear = output_thread * 8 + pass * T::BLOCK_SIZE * 8;
        const int output_row = linear / T::B_N;
        const int output_col = linear % T::B_N;
        const auto value = load<8>(s_c, output_row * C_PITCH + output_col);
        // Keep the original bounded g_c resource. Check missing rows before
        // forming a wide-N offset, as in the established 9000 BF16 epilogue.
        if (row + output_row < kargs.m)
            store<8>(g_c, value, output_row * kargs.stride_c + output_col,
                     0, opus::number<2>{});
    });

}
#endif
