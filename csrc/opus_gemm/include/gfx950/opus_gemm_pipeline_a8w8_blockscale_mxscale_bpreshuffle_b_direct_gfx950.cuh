// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_b_direct_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_b_direct_detail {
template <class Traits>
__device__ __forceinline__ void
bulk_panel_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(OPUS_BP_FIXED_M) && defined(OPUS_BP_FIXED_N) && defined(OPUS_BP_FIXED_K)
    opus_gemm_blockscale_mxscale_bpreshuffle_apply_fixed_args(args);
#endif
    using namespace opus;
    using T = opus::remove_cvref_t<Traits>;
    using Index = unsigned;
    constexpr int WaveColumns = T::T_N;

    // Prologue: runtime tile descriptors, operand rings, and supplied scale panels.
    constexpr int BM = T::B_M, BN = T::B_N, NB = T::STAGES, BD = T::B_AHEAD, BS = T::B_SETS,
                  P = T::PANELS;
    constexpr int MR = T::MR, NR = T::NR, K = T::K, SC = T::SCALE_COLS, BK = T::B_K, KS = BK / 128;
    const int M = args.m, N = args.n;
    const int split = T::SPLIT_K > 1 ? block_id_z() : 0;
    using D_C = std::conditional_t<T::SPLIT_K == 1, bf16_t, float>;
    const int stride_a = T::CONTIGUOUS ? T::TOTAL_K : args.stride_a, stride_b = T::CONTIGUOUS ? T::TOTAL_K : args.stride_b, stride_c = args.stride_c;
    static_assert((BK == 128 || BK == 256) && SC == K / 128);
    static_assert(T::T_M == 1 && T::T_N * 64 == T::BLOCK_SIZE);
    static_assert(T::MR * 16 == T::B_M && T::NR * T::T_N * 16 == T::B_N);
    const Index tid = thread_id_x(), lane = tid % 64, lane16 = lane % 16, kg = lane / 16;
    const Index wave = __builtin_amdgcn_readfirstlane(tid / 64), wn = wave;
    if constexpr(requires { T::THREAD_RANGE_KNOWN; }) {
        __builtin_assume(static_cast<unsigned>(tid)<T::BLOCK_SIZE);
        __builtin_assume(static_cast<unsigned>(wave)<T::BLOCK_SIZE/64);
    }
    const unsigned pid = block_id_x(), gm = (M + BM - 1) / BM, gn = N / BN, total = gm * gn;
    const unsigned tile = T::XCD_ORDER ? pid % 8 * ((total + 7) / 8) + pid / 8 : pid;
    if constexpr (T::XCD_ORDER)
        if (tile >= total)
            return;
    const int row = T::GRID_2D ? block_id_y() * BM : (M <= BM ? 0 : (tile / gn) * BM);
    const int col = T::GRID_2D ? block_id_x() * BN : (M <= BM ? tile * BN : (tile % gn) * BN);
    int valid_rows = M % BM == 0 ? BM : (M <= BM ? M : (row == (gm - 1) * BM ? M % BM : BM));
    if constexpr(requires { T::SIMPLE_TAIL_ROWS; })
        valid_rows = M - row < BM ? M - row : BM;
    auto ga = make_gmem(reinterpret_cast<const fp8_t *>(args.ptr_a) + row * stride_a + split * K,
                        static_cast<unsigned>(valid_rows * stride_a - split * K));
    auto gb = make_gmem(reinterpret_cast<const fp8_t *>(args.ptr_b) + col * stride_b + split * K * 16,
                        static_cast<unsigned>(BN * stride_b - split * K * 16));
    auto gsa = make_gmem(reinterpret_cast<const unsigned char *>(args.ptr_sfa) + split * SC * args.stride_sfa,
                         static_cast<unsigned>((SC - 1) * args.stride_sfa + M));
    auto gsb = make_gmem(reinterpret_cast<const unsigned char *>(args.ptr_sfb) + split * SC,
                         static_cast<unsigned>((N / 128 - 1) * args.stride_sfb + SC));
    auto gc = make_gmem(reinterpret_cast<D_C *>(args.ptr_c) +
                            static_cast<int64_t>(split) * M * stride_c +
                            static_cast<int64_t>(row) * stride_c + col,
                        static_cast<unsigned>(((valid_rows - 1) * stride_c + BN) * sizeof(D_C)));
    alignas(16) __shared__ unsigned char arena[T::LDS_BYTES];
    auto sa = make_smem(reinterpret_cast<fp8_t *>(arena));
    auto ss = make_smem(arena);
    using Base =
        decltype(make_mfma<fp8_t, fp8_t, fp32_t>(16_I, 16_I, 128_I, mfma_adaptor_swap_ab{}));
    array<typename Base::vtype_c, MR * NR> c{};
    array<array<array<typename Base::vtype_b, NR>, KS>, BS> bregs;
    auto issue_b = [&](auto bi, int kt) {
        constexpr int set = decltype(bi)::value;
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<NR>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const int br = wn * (BN / WaveColumns) + n * 16,
                          off = ks * 2048 + kg * 256 + lane16 * 16;
                set_slice(bregs[set][ks][n],
                          load<16>(gb, off, br * stride_b + kt * (BK * 16), number<T::B_CACHE>{}), 0_I,
                          16_I);
                set_slice(bregs[set][ks][n],
                          load<16>(gb, off + 1024, br * stride_b + kt * (BK * 16), number<T::B_CACHE>{}),
                          16_I, 32_I);
            });
        });
        __builtin_amdgcn_sched_barrier(0);
    };
    auto issue_a = [&](int slot, int kt) {
        static_for<T::A_COPIES>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            const int lin = (i * T::BLOCK_SIZE + tid) * 16, r = lin / BK,
                      kc = (lin % BK) ^ ((r % (BK / 16)) * 16);
            auto *dst = reinterpret_cast<OPUS_LDS_ADDR void *>(reinterpret_cast<__UINTPTR_TYPE__>(
                arena + slot * T::STAGE_BYTES + (i * T::BLOCK_SIZE + wave * 64) * 16));
            __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
                ga.cached_rsrc, dst, 16, r * stride_a + kc, kt * BK, 0, T::CACHECTL_A);
        });
        __builtin_amdgcn_asyncmark();
        __builtin_amdgcn_sched_barrier(0);
    };
    static_for<T::INITIAL_B>([&](auto i) { issue_b(i, decltype(i)::value); });
    issue_a(0, 0);
    if constexpr (T::BYTE_SCALE) {
        static_for<T::SA_COPIES>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            const int index = i * T::BLOCK_SIZE + tid;
            const int sr = index % T::SA_ROWS, scol = index / T::SA_ROWS;
            int source = sr < BM && row + sr < M && scol < SC
                ? scol * args.stride_sfa + row + sr : (SC - 1) * args.stride_sfa + M;
            asm volatile("" : "+v"(source));
            auto *dst = reinterpret_cast<OPUS_LDS_ADDR void *>(reinterpret_cast<__UINTPTR_TYPE__>(
                arena + T::OFF_SA + (i * T::BLOCK_SIZE + wave * 64) * 4));
            __builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc, dst, 1, source, 0, 0, 0);
        });
    } else {
        static_for<T::SA_COPIES>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            const int index = (i * T::BLOCK_SIZE + tid) * 16, sr = index % T::SA_ROWS,
                      scol = index / T::SA_ROWS, sc = scol < SC ? scol : SC - 1;
            int source = sr < BM && row + sr < M ? sc * args.stride_sfa + row + sr
                                                 : (SC - 1) * args.stride_sfa + M;
            asm volatile("" : "+v"(source));
            auto *dst = reinterpret_cast<OPUS_LDS_ADDR void *>(reinterpret_cast<__UINTPTR_TYPE__>(
                arena + T::OFF_SA + (i * T::BLOCK_SIZE + wave * 64) * 16));
            __builtin_amdgcn_raw_ptr_buffer_load_async_lds(gsa.cached_rsrc, dst, 16, source, 0, 0, 0);
        });
    }
    auto *sbdst = reinterpret_cast<OPUS_LDS_ADDR void *>(
        reinterpret_cast<__UINTPTR_TYPE__>(arena + T::OFF_SB + wave * 64 * 4));
    if constexpr (T::SB_SLOTS) {
        const int index = tid % (T::SB_ROWS * SC), sr = index / SC, sc = index % SC;
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
            gsb.cached_rsrc, sbdst, 1, (col / 128 + sr) * args.stride_sfb + sc, 0, 0, 0);
    } else {
        const int index = tid % (T::SB_ROWS * (SC / 4)), sr = index / (SC / 4),
                  word = index % (SC / 4);
        __builtin_amdgcn_raw_ptr_buffer_load_async_lds(
            gsb.cached_rsrc, sbdst, 4, (col / 128 + sr) * args.stride_sfb + word * 4, 0, 0, 0);
    }
    static_for<T::INITIAL_A - 1>(
        [&](auto i) { issue_a(decltype(i)::value + 1, decltype(i)::value + 1); });
    // Main loop: wait for the A slot, read operands, refill, and compute.
    auto step = [&](int t, int slot, auto bi, auto wai, auto wbi, auto fai, auto fbi) {
        constexpr int set = decltype(bi)::value, wa = decltype(wai)::value,
                      wb = decltype(wbi)::value;
        constexpr bool future_a = decltype(fai)::value, future_b = decltype(fbi)::value;
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wa > 63 ? 63 : wa)>{});
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        if constexpr (future_b)
            issue_b(number<(set + BD) % BS>{}, t + BD);
        array<array<typename Base::vtype_a, MR>, KS> af;
        array<array<unsigned, MR>, KS> as;
        array<array<unsigned, NR>, KS> bs;
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            const int scale_col = t * KS + ks;
            static_for<NR>([&](auto ni) {
                constexpr int nr = decltype(ni)::value;
                const int group_n = (wn * (BN / WaveColumns) + nr * 16) / 128;
                bs[ks][nr] = load<1>(ss, T::OFF_SB + (group_n * SC + scale_col) * (T::SB_SLOTS ? 4 : 1))[0];
            });
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int ar = m * 16 + lane16;
                const int lo =
                    slot * T::STAGE_BYTES + ar * BK + (((ks * 8 + kg) ^ (ar % (BK / 16))) * 16);
                const int hi =
                    slot * T::STAGE_BYTES + ar * BK + (((ks * 8 + kg + 4) ^ (ar % (BK / 16))) * 16);
                if constexpr (T::PIN_A) {
                    array<int,2> offsets;
                    offsets[0] = lo; offsets[1] = hi;
                    opus_gemm_bpreshuffle_ring_pin_detail::
                        load_operand_fragment_pinned<16,0>(number<(ks * MR + m) * 8>{}, af[ks][m], sa, offsets);
                } else {
                    set_slice(af[ks][m], load<16>(sa, lo), 0_I, 16_I);
                    set_slice(af[ks][m], load<16>(sa, hi), 16_I, 32_I);
                }
                as[ks][m] = load<1>(ss, T::OFF_SA + (scale_col * T::SA_ROWS + ar) * T::SA_SCALE)[0];
            });
        });
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (future_a)
            issue_a(slot == 0 ? NB - 1 : slot - 1, t + NB - 1);
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_vmcnt(number<(wb > 63 ? 63 : wb)>{});
        s_waitcnt_lgkmcnt(0_I);
        static_for<KS>([&](auto ki) {
            constexpr int ks = decltype(ki)::value;
            static_for<MR>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                static_for<NR>([&](auto ni) {
                    constexpr int n = decltype(ni)::value, ci = m * NR + n;
                    if constexpr (T::PIN_C) {
                        [[clang::amdgpu_pin_agpr(128 + ci * 4)]] c[ci] = c[ci];
                        [[clang::amdgpu_pin_agpr(128 + ci * 4)]] c[ci] = Base{}(
                            af[ks][m], bregs[set][ks][n], c[ci], static_cast<int>(as[ks][m]),
                            static_cast<int>(bs[ks][n]), 0_I, 0_I);
                    } else {
                        c[ci] = Base{}(af[ks][m], bregs[set][ks][n], c[ci], static_cast<int>(as[ks][m]),
                                       static_cast<int>(bs[ks][n]), 0_I, 0_I);
                    }
                });
            });
        });
        if constexpr (!T::C_AGPR && !T::PIN_C)
            static_for<MR * NR>([&](auto ci) {
                auto &v = c[decltype(ci)::value];
                asm volatile("" : "+v"(v));
            });
        __builtin_amdgcn_sched_barrier(0);
    };
    constexpr int START = T::STEADY < NB - 1 ? T::STEADY : NB - 1;
    static_for<START>([&](auto ti) {
        constexpr int t = decltype(ti)::value;
        step(t, t % NB, number<t % BS>{}, number<T::wait_a(t)>{}, number<T::wait_b(t)>{},
             number<true>{}, number<(t + BD < P)>{});
    });
    constexpr int PHASES = []() constexpr {
        int a = NB, b = BS;
        while (b) {
            int r = a % b;
            a = b;
            b = r;
        }
        return NB / a * BS;
    }();
    constexpr int GROUPS = (T::STEADY - START) / PHASES, REGULAR_END = START + GROUPS * PHASES;
    if constexpr (GROUPS > 0) {
#pragma clang loop unroll(disable)
        for (int group = 0; group < GROUPS; ++group) {
            static_for<PHASES>([&](auto phase) {
                constexpr int ph = decltype(phase)::value, t0 = START + ph;
                step(START + group * PHASES + ph, t0 % NB, number<t0 % BS>{},
                     number<T::wait_a(t0)>{}, number<T::wait_b(t0)>{}, number<true>{},
                     number<true>{});
            });
        }
    }
    static_for<P - REGULAR_END>([&](auto ti) {
        constexpr int t = REGULAR_END + decltype(ti)::value;
        step(t, t % NB, number<t % BS>{}, number<T::wait_a(t)>{}, number<T::wait_b(t)>{},
             number<(t + NB - 1 < P)>{}, number<(t + BD < P)>{});
    });
    if constexpr (T::SPLIT_K > 1) {
        static_for<MR>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            const int rr = m * 16 + lane16;
            static_for<NR>([&](auto ni) {
                constexpr int n = decltype(ni)::value, ci = m * NR + n;
                const int nn = wn * (BN / WaveColumns) + n * 16 + kg * 4;
                store<4>(gc, c[ci], rr * stride_c + nn, 0, number<T::STORE_CACHE>{});
            });
        });
    } else if constexpr (T::OUTPUT == 2) {
        opus_gemm_blockscale_mxscale_bpreshuffle_register_prefetch_mfma16_output::
            store_mfma16_packed<T, true>(c, gc, lane, 0, wn, stride_c);
    } else {
    // Epilogue: reduce the output layout to contiguous bf16 stores.
    auto sc = make_smem(reinterpret_cast<bf16_t *>(arena));
    constexpr int MAX_ROWS = BM;
    static_for<(MAX_ROWS + T::C_ROWS - 1) / T::C_ROWS>([&](auto parti) {
        constexpr int part = decltype(parti)::value;
        constexpr int PART_ROWS =
            MAX_ROWS - part * T::C_ROWS < T::C_ROWS ? MAX_ROWS - part * T::C_ROWS : T::C_ROWS;
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        static_for<MR>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            if constexpr (m * 16 / T::C_ROWS == part && m * 16 < MAX_ROWS)
                static_for<NR>([&](auto ni) {
                    constexpr int n = decltype(ni)::value, ci = m * NR + n;
                    const int rr = m * 16 + lane16, nn = wn * (BN / WaveColumns) + n * 16 + kg * 4;
                    store<4>(sc, cast<bf16_t>(c[ci]), (rr - part * T::C_ROWS) * (BN + 8) + nn);
                });
        });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        constexpr int CHUNKS = (PART_ROWS * (BN / 8) + T::BLOCK_SIZE - 1) / T::BLOCK_SIZE;
        array<vector_t<bf16_t, 8>, CHUNKS> output;
        static_for<CHUNKS>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            const int index = i * T::BLOCK_SIZE + tid, rr = index / (BN / 8),
                      nn = index % (BN / 8) * 8;
            if constexpr (PART_ROWS * (BN / 8) % T::BLOCK_SIZE == 0)
                output[i] = load<8>(sc, rr * (BN + 8) + nn);
            else if (rr < PART_ROWS)
                output[i] = load<8>(sc, rr * (BN + 8) + nn);
        });
        __builtin_amdgcn_sched_barrier(0);
        s_waitcnt_lgkmcnt(0_I);
        static_for<CHUNKS>([&](auto ii) {
            constexpr int i = decltype(ii)::value;
            const int index = i * T::BLOCK_SIZE + tid, rr = index / (BN / 8),
                      nn = index % (BN / 8) * 8;
            if (rr < PART_ROWS && part * T::C_ROWS + rr < valid_rows)
                store<8>(gc, output[i], (part * T::C_ROWS + rr) * stride_c + nn, 0,
                         number<T::STORE_CACHE>{});
        });
    });
    }
    s_waitcnt_vmcnt(0_I);
}
} // namespace opus_gemm_b_direct_detail
#endif

template <class Traits>
__global__ __launch_bounds__(
    Traits::BLOCK_SIZE,
    Traits::
        WG_PER_CU) void opus_gemm_blockscale_mxscale_bpreshuffle_b_direct_bulk_panel_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950
                                                                                                args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    opus_gemm_b_direct_detail::bulk_panel_body<Traits>(args);
#endif
}
