// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <opus/hip_minimal.hpp>
#include <opus/opus.hpp>
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"
#include "opus_gemm_mxscale_bpreshuffle_small_output_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void gemm_a8w8_mxfp8_scale_small_register_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE, 1)
void gemm_a8w8_mxfp8_scale_small_register_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args) {
    using namespace opus;
    const int lane = thread_id_x() % 64;
    const int wave = __builtin_amdgcn_readfirstlane(thread_id_x() / 64);
    const int wm = wave % T::T_M;
    const int wn = (wave / T::T_M) % T::T_N;
    const int wk = wave / (T::T_M * T::T_N);
    const int row = block_id_y() * T::B_M, col = block_id_x() * T::B_N;
    const int total_tiles = (T::FIXED_K ? T::FIXED_K : args.k) / T::W_K;
    const int tiles_per_wave = total_tiles / T::WAVE_K;
    const int extra_tiles = total_tiles % T::WAVE_K;
    const int loops = tiles_per_wave + (wk < extra_tiles);
    const int tile_begin = wk * tiles_per_wave + (wk < extra_tiles ? wk : extra_tiles);
    const int scale_groups = args.k / 128;
    auto ga = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a) + row * args.stride_a,
                        static_cast<unsigned>((args.m - row) * args.stride_a));
    auto gb = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) + col * args.stride_b,
                        T::N_TAIL ? static_cast<unsigned>((args.n - col) * args.stride_b) : 0xffffffffu);
    auto gsa = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa),
                         static_cast<unsigned>(args.m * scale_groups));
    auto gsb = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb),
                         T::N_TAIL ? static_cast<unsigned>(args.n / 128 * scale_groups) : 0xffffffffu);
    auto gc = make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c) + row * args.stride_c + col,
                        static_cast<unsigned>(((args.m - row) * args.stride_c - col) * 2));
    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(
        seq<T::E_M, T::E_N, 1>{}, seq<T::T_M, T::T_N, 1>{},
        seq<T::W_M, T::W_N, T::W_K>{}, mfma_adaptor_swap_ab{});
    using Base = typename decltype(mma)::MMA;
    constexpr int elem_c = T::W_M * T::W_N / 64;
    struct operands {
        array<typename Base::vtype_a, T::E_M> a;
        array<typename Base::vtype_b, T::E_N> b;
        array<unsigned, T::E_M> sfa;
        array<unsigned, T::E_N> sfb;
    };
    array<operands, T::PREFETCH> queue;
    array<typename Base::vtype_c, T::E_M * T::E_N> c{};

    // The physical N16/K16 B layout feeds MFMA directly; no matrix LDS is needed.
    auto prefetch = [&](auto stage, int kt) {
        auto& q = queue[decltype(stage)::value];
        if constexpr (T::WAVE_K > 1) kt += tile_begin;
        if constexpr (std::is_same_v<T, opus_gemm_small_register_traits_gfx950<
                16, 48, 1, 1, 4, 4, 4, 3, 7168, true, false>>) {
            // Issue the same raw scales first; no operand is consumed until
            // the original compute queues/waits and K-wave reduction below.
            static_for<T::E_M>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int r = (m * T::T_M + wm) * T::W_M + lane % T::W_M;
                const int sf_offset = r + row < args.m ? (kt * T::W_K / 128) * args.stride_sfa + row + r : -1;
                q.sfa[m] = load<1>(gsa, sf_offset)[0];
            });
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const int nr = (n * T::T_N + wn) * T::W_N;
                q.sfb[n] = load<1>(gsb, ((col + nr) / 128) * args.stride_sfb + kt * T::W_K / 128)[0];
            });
            __builtin_amdgcn_sched_barrier(0);
            static_for<T::E_M>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int r = (m * T::T_M + wm) * T::W_M + lane % T::W_M;
                const int offset = r * args.stride_a + (lane / T::W_M) * 16;
                set_slice(q.a[m], load<16>(ga, offset, kt * T::W_K), 0_I, 16_I);
                set_slice(q.a[m], load<16>(ga, offset + T::W_K / 2, kt * T::W_K), 16_I, 32_I);
            });
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const int nr = (n * T::T_N + wn) * T::W_N;
                const int offset = nr * args.stride_b + lane * 16;
                set_slice(q.b[n], load<16>(gb, offset, kt * T::W_K * 16, number<T::B_CACHE>{}), 0_I, 16_I);
                set_slice(q.b[n], load<16>(gb, offset + T::W_K * 8, kt * T::W_K * 16, number<T::B_CACHE>{}), 16_I, 32_I);
            });
        } else {
            static_for<T::E_M>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int r = (m * T::T_M + wm) * T::W_M + lane % T::W_M;
                const int offset = r * args.stride_a + (lane / T::W_M) * 16;
                set_slice(q.a[m], load<16>(ga, offset, kt * T::W_K), 0_I, 16_I);
                set_slice(q.a[m], load<16>(ga, offset + T::W_K / 2, kt * T::W_K), 16_I, 32_I);
                const int sf_offset = r + row < args.m ? (kt * T::W_K / 128) * args.stride_sfa + row + r : -1;
                q.sfa[m] = load<1>(gsa, sf_offset)[0];
            });
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                const int nr = (n * T::T_N + wn) * T::W_N;
                const int offset = nr * args.stride_b + lane * 16;
                set_slice(q.b[n], load<16>(gb, offset, kt * T::W_K * 16, number<T::B_CACHE>{}), 0_I, 16_I);
                set_slice(q.b[n], load<16>(gb, offset + T::W_K * 8, kt * T::W_K * 16, number<T::B_CACHE>{}), 16_I, 32_I);
                if constexpr (!T::REUSE_B_SCALE) {
                    q.sfb[n] = load<1>(gsb, ((col + nr) / 128) * args.stride_sfb + kt * T::W_K / 128)[0];
                } else if constexpr (n == 0) {
                    q.sfb[0] = load<1>(gsb, (col / 128) * args.stride_sfb + kt * T::W_K / 128)[0];
                }
            });
        }
    };
    auto compute = [&](auto stage) {
        auto& q = queue[decltype(stage)::value];
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value, ci = m * T::E_N + n;
                c[ci] = Base{}(q.a[m], q.b[n], c[ci],
                               static_cast<int>(q.sfa[m]),
                               static_cast<int>(q.sfb[T::REUSE_B_SCALE ? 0 : n]), 0_I, 0_I);
            });
        });
    };
    static_for<T::PREFETCH>([&](auto i) {
        if (decltype(i)::value < loops) prefetch(i, decltype(i)::value);
    });
    // Full queue groups overlap compute with the next prefetch. Only the
    // prologue and tail need bounds checks, including waves with no K tiles.
    const int full = loops / T::PREFETCH, tail = loops % T::PREFETCH;
    auto advance = [&](int group) {
        static_for<T::PREFETCH>([&](auto i) {
            compute(i);
            prefetch(i, (group + 1) * T::PREFETCH + decltype(i)::value);
        });
    };
    if constexpr (T::FIXED_K) {
        #pragma clang loop unroll(full)
        for (int group = 0; group + 1 < full; ++group) advance(group);
    } else {
        // Runtime K keeps one queue group to limit VGPR pressure.
        #pragma clang loop unroll(disable)
        for (int group = 0; group + 1 < full; ++group) advance(group);
    }
    if (full > 0) {
        static_for<T::PREFETCH>([&](auto i) {
            compute(i);
            if (decltype(i)::value < tail)
                prefetch(i, full * T::PREFETCH + decltype(i)::value);
        });
    }
    static_for<T::PREFETCH>([&](auto i) {
        if (decltype(i)::value < tail) compute(i);
    });

    if constexpr (T::OUTPUT == 4) {
        // Each final fragment belongs to one K wave. All partials stay in LDS
        // until that wave sums them and performs the sole BF16 conversion.
        constexpr int elems = T::B_M * T::B_N;
        alignas(64) __shared__ float partials[T::WAVE_K * elems];
        auto shared = make_smem(partials);
        const int offset = ((wave % (T::T_M * T::T_N)) * T::E_M * T::E_N * 64 + lane) * elem_c;
        static_for<T::E_M * T::E_N>([&](auto i) {
            store<elem_c>(shared, c[decltype(i)::value],
                wk * elems + offset + decltype(i)::value * 64 * elem_c);
        });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        const auto pc = opus::make_tuple(wm, lane % mma.grpn_c, wn, lane / mma.grpn_c);
        const auto uc = partition_layout_c<4>(mma, opus::make_tuple(args.stride_c, 1_I), pc);
        const auto offsets = layout_to_offsets<4>(uc);
        static_for<T::E_M * T::E_N>([&](auto i) {
            constexpr int fragment = decltype(i)::value;
            if (wk == fragment % T::WAVE_K) {
                array<typename Base::vtype_c, T::WAVE_K> values;
                static_for<T::WAVE_K>([&](auto sk) {
                    values[decltype(sk)::value] = load<elem_c>(shared,
                        decltype(sk)::value * elems + offset + fragment * 64 * elem_c);
                });
                typename Base::vtype_c sum{};
                static_for<T::WAVE_K>([&](auto sk) { sum += values[decltype(sk)::value]; });
                if constexpr (T::N_TAIL) {
                    const int first_col = col + (fragment % T::E_N * T::T_N + wn) * T::W_N;
                    if (first_col < args.n) store<4>(gc, cast<bf16_t>(sum), offsets[fragment]);
                } else {
                    store<4>(gc, cast<bf16_t>(sum), offsets[fragment]);
                }
            }
        });
    } else {
        // Small grids can partition K between waves. Accumulate their FP32 results
        // once in LDS before BF16 conversion, without a workspace or extra launch.
        if constexpr (T::WAVE_K > 1) {
            constexpr int elems = T::B_M * T::B_N;
            alignas(64) __shared__ float partials[(T::WAVE_K - 1) * elems];
            auto shared = make_smem(partials);
            const int offset = ((wave % (T::T_M * T::T_N)) * T::E_M * T::E_N * 64 + lane) * elem_c;
            if (wk > 0) {
                static_for<T::E_M * T::E_N>([&](auto i) {
                    store<elem_c>(shared, c[decltype(i)::value],
                                  (wk - 1) * elems + offset + decltype(i)::value * 64 * elem_c);
                });
            }
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
            if (wk > 0) return;
            static_for<T::WAVE_K - 1>([&](auto sk) {
                static_for<T::E_M * T::E_N>([&](auto i) {
                    c[decltype(i)::value] += load<elem_c>(shared,
                        decltype(sk)::value * elems + offset + decltype(i)::value * 64 * elem_c);
                });
            });
        }
        if constexpr (T::OUTPUT == 3) {
            opus_gemm_small_output::store_mfma16_packed<T>(c, gc, lane, wm, wn, args.stride_c);
        } else {
            const auto pc = opus::make_tuple(wm, lane % mma.grpn_c, wn, lane / mma.grpn_c);
            const auto uc = partition_layout_c<4>(mma, opus::make_tuple(args.stride_c, 1_I), pc);
            const auto offsets = layout_to_offsets<4>(uc);
            static_for<T::E_M * T::E_N>([&](auto i) {
                store<4>(gc, cast<bf16_t>(c[decltype(i)::value]), offsets[decltype(i)::value]);
            });
        }
    }
}
#endif
