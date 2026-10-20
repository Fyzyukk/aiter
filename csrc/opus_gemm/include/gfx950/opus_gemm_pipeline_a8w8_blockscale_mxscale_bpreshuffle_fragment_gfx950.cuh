// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "../opus_gemm_utils.cuh"
#include "opus_gemm_kargs_a8w8_mxscale_bpreshuffle_gfx950.cuh"
#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_fragment_gfx950.cuh"

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_fragment_detail {
using namespace opus;

// The memory layouts describe the two 16-byte halves of a scaled-MFMA
// operand. No intermediate LDS representation or workgroup barrier is needed.
template<class T> struct fragment_engine {
    using MMA = decltype(make_mfma<fp8_t, fp8_t, fp32_t>(
        seq<16, 16, 128>{}, mfma_adaptor_swap_ab{}));
    struct packet {
        array<typename MMA::vtype_a, T::E_M> a;
        array<typename MMA::vtype_b, T::E_N> b;
        array<unsigned, T::E_M> sa;
        array<unsigned, T::N_SCALES> sb;
    };
    using accumulator = array<typename MMA::vtype_c, T::E_M * T::E_N>;

    template<int Count, class Scales>
    __device__ static auto scale_words(const Scales& scales) {
        constexpr int bytes = T::PACK_SCALES ? 4 : 1;
        array<unsigned, (Count + bytes - 1) / bytes> result{};
        static_for<Count>([&](auto i) {
            constexpr int j = decltype(i)::value;
            result[j / bytes] |= scales[j] << ((j % bytes) * 8);
        });
        return result;
    }

    template<bool LoadScales = true, class GA, class GB, class GSA, class GSB>
    __device__ static packet fetch(GA& ga, GB& gb, GSA& gsa, GSB& gsb,
                                  int kblock, int stride_a, int stride_b,
                                  int stride_sa, int stride_sb,
                                  int row, int col, int shape_m,
                                  int lane, int wm, int wn) {
        packet p;
        if constexpr(!LoadScales) {
            p.sa = {};
            p.sb = {};
        }
        const int mr = wm * T::E_M * 16 + lane % 16;
        const int kg = lane / 16;
        const auto halves_a = make_layout<16>(tuple{2_I, 16_I}, tuple{64_I, 1_I},
                                             tuple{underscore{}, underscore{}});
        const auto halves_b = make_layout<16>(tuple{2_I, 16_I}, tuple{1024_I, 1_I},
                                             tuple{underscore{}, underscore{}});
        static_for<T::E_M>([&](auto i) {
            constexpr int m = decltype(i)::value;
            const int r = mr + m * 16;
            p.a[m] = load<16>(ga, halves_a + (r * stride_a + kblock * 128 + kg * 16),
                             0, number<T::CACHE_A>{});
            // A is zero outside M. Bound scale addressing separately so a tail
            // row cannot accidentally consume the next K block's exponent.
            if constexpr(LoadScales) {
                const int sr = row + r < shape_m ? row + r : row;
                p.sa[m] = load<1>(gsa, kblock * stride_sa + sr)[0];
            }
        });
        static_for<T::E_N>([&](auto i) {
            constexpr int n = decltype(i)::value;
            const int ng = n * T::T_N + wn;
            p.b[n] = load<16>(gb, halves_b + (ng * 16 * stride_b + kblock * 2048 + lane * 16),
                             0, number<T::CACHE_B>{});
        });
        if constexpr(LoadScales) {
            static_for<T::N_SCALES>([&](auto i) {
                constexpr int s = decltype(i)::value;
                p.sb[s] = load<1>(gsb, (col / 128 + s) * stride_sb + kblock)[0];
            });
        }
        return p;
    }

    __device__ static void contract(const packet& p, accumulator& c) {
        const auto sa = [&] { if constexpr(T::SCALES_PREPACKED) return p.sa; else return scale_words<T::E_M>(p.sa); }();
        const auto sb = [&] { if constexpr(T::SCALES_PREPACKED) return p.sb; else return scale_words<T::N_SCALES>(p.sb); }();
        constexpr int bytes = T::PACK_SCALES ? 4 : 1;
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value, ns = n / T::N_PER_SCALE;
                c[m * T::E_N + n] = MMA{}(p.a[m], p.b[n], c[m * T::E_N + n],
                    int(sa[m / bytes]), int(sb[ns / bytes]),
                    number<m % bytes>{}, number<ns % bytes>{});
            });
        });
    }
};

// Grouped DMA must use the asynchronous intrinsic. The legacy OPUS
// async_load spelling emits raw_ptr_buffer_load_lds, which is not tracked by
// asyncmark. Keep this adapter local to the pipeline's transport policy.
template<int Bytes, class Memory>
__device__ void grouped_copy(Memory& source, void* destination, int offset) {
    auto* dst = reinterpret_cast<OPUS_LDS_ADDR void*>(reinterpret_cast<__UINTPTR_TYPE__>(destination));
    __builtin_amdgcn_raw_ptr_buffer_load_async_lds(source.cached_rsrc, dst, Bytes, offset, 0, 0, 0);
}

// A shared-memory transport for the same packet contraction. A complete K
// panel is consumed before its slot is released. The other panel may remain
// in flight throughout those contractions; there is no global B register ring.
template<class T>
__device__ void shared_body(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
    using Engine = fragment_engine<T>;
    const int row = block_id_y() * T::B_M, col = block_id_x() * T::B_N;
    const int tid = thread_id_x(), lane = tid % 64;
    __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wm = wave / T::T_N, wn = wave % T::T_N;
    const int rows = args.m - row < T::B_M ? args.m - row : T::B_M;
    const int sc = args.k / 128;
    auto ga = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a) + int64_t(row) * args.stride_a,
                       rows * args.stride_a);
    auto gb = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) + int64_t(col) * args.stride_b,
                       T::B_N * args.stride_b);
    auto gsa = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa), args.m * sc);
    auto gsb = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb), (args.n / 128) * sc);
    extern __shared__ __attribute__((aligned(16))) unsigned char arena[];
    auto sm = make_smem(reinterpret_cast<fp8_t*>(arena));
    auto ss = make_smem(arena);
    const int sb_offset = T::SCALE_A_OFFSET + T::scale_a_bytes(args.k);
    if constexpr(T::SCALES_PREPACKED) {
        // Transpose the small scale tile once into words owned by an MFMA
        // lane. The inner K loop then reads one dword for four M repeats.
        const int copies = (T::B_M * sc + T::BLOCK_SIZE * 4 - 1) / (T::BLOCK_SIZE * 4);
        for (int i = 0; i < copies; ++i) {
            const int index = (i * T::BLOCK_SIZE + tid) * 4;
            const int r = index % T::B_M, ks = index / T::B_M;
            if (ks < sc) {
                const int source = row + r < args.m ? ks * args.m + row + r : args.m * sc;
                auto values = load<4>(gsa, source);
                const int wave_m = r / (T::E_M * 16), repeat = (r / 16) % T::E_M;
                const int offset = T::SCALE_A_OFFSET + ks * T::SCALE_M_BYTES +
                    (wave_m * T::M_WORDS + repeat / 4) * 64 + (r % 16) * 4 + repeat % 4;
                static_for<4>([&](auto j) {
                    store<1>(ss, slice(values, number<decltype(j)::value>{}, number<decltype(j)::value + 1>{}), offset + decltype(j)::value * 4);
                });
            }
        }
        for (int ks = tid; ks < sc; ks += T::BLOCK_SIZE) {
            unsigned word = 0;
            static_for<T::N_SCALES>([&](auto ni) {
                constexpr int ns = decltype(ni)::value;
                word |= unsigned(load<1>(gsb, (col / 128 + ns) * args.stride_sfb + ks)[0]) << (ns * 8);
            });
            auto words = make_smem(reinterpret_cast<unsigned*>(arena));
            store<1>(words, make_repeated_vector(word, 1_I), sb_offset / 4 + ks);
        }
        s_waitcnt_lgkmcnt(0_I);
    } else {
    // A's column-major exponents are copied as compact dwords. Host eligibility
    // requires M % 4 == 0; invalid whole dwords are ignored by the A/C bounds.
    for (int i = 0; i < T::scale_a_bytes(args.k) / T::SCALE_A_DMA; ++i) {
        const int index = (i * T::BLOCK_SIZE + tid) * 4;
        const int r = index % T::B_M, ks = index / T::B_M;
        const int source = ks < sc && row + r < args.m ? ks * args.m + row + r : args.m * sc;
        grouped_copy<4>(gsa, arena + T::SCALE_A_OFFSET + (i * T::BLOCK_SIZE + wave * 64) * 4,
                      source);
    }
    for (int i = 0; i < T::scale_b_bytes(args.k) / (T::BLOCK_SIZE * 4); ++i) {
        const int index = i * T::BLOCK_SIZE + tid;
        const int ng = index / sc, ks = index % sc;
        const int source = ng < (T::B_N + 127) / 128 ? (col / 128 + ng) * args.stride_sfb + ks : (args.n / 128) * sc;
        grouped_copy<1>(gsb, arena + sb_offset + (i * T::BLOCK_SIZE + wave * 64) * 4, source);
    }
    }
    using BFragment = typename Engine::MMA::vtype_b;
    array<array<array<BFragment, T::E_N>, T::STAGE_K / 128>, 2> b_queue;
    auto issue = [&](auto slot_c, int panel) {
        constexpr int slot = decltype(slot_c)::value;
        static_for<T::A_BYTES / T::DMA>([&](auto i) {
            const int lin = (decltype(i)::value * T::BLOCK_SIZE + tid) * 16;
            const int r = lin / T::STAGE_K;
            const int kc = (lin % T::STAGE_K) ^ ((r % (T::STAGE_K / 16)) * 16);
            grouped_copy<16>(ga, arena + slot * T::PANEL_BYTES + (decltype(i)::value * T::BLOCK_SIZE + wave * 64) * 16,
                           r * args.stride_a + panel * T::STAGE_K + kc);
        });
        static_for<T::B_BYTES / T::DMA>([&](auto i) {
            const int lin = (decltype(i)::value * T::BLOCK_SIZE + tid) * 16;
            const int ng = lin / (16 * T::STAGE_K), within = lin % (16 * T::STAGE_K);
            grouped_copy<16>(gb, arena + slot * T::PANEL_BYTES + T::A_BYTES + (decltype(i)::value * T::BLOCK_SIZE + wave * 64) * 16,
                           ng * 16 * args.stride_b + panel * T::STAGE_K * 16 + within);
        });
        if constexpr(T::REGISTER_B) {
            const auto halves = make_layout<16>(tuple{2_I,16_I}, tuple{1024_I,1_I},
                                                 tuple{underscore{},underscore{}});
            static_for<T::STAGE_K / 128>([&](auto ki) {
                constexpr int ks = decltype(ki)::value;
                static_for<T::E_N>([&](auto ni) {
                    constexpr int n = decltype(ni)::value;
                    const int offset = (n * T::T_N + wn) * 16 * args.stride_b +
                        panel * T::STAGE_K * 16 + ks * 2048 + lane * 16;
                    b_queue[slot][ks][n] = load<16>(gb, halves + offset);
                });
            });
        }
        __builtin_amdgcn_asyncmark();
    };
    auto read = [&](auto slot_c, int panel, auto ki) {
        constexpr int slot = decltype(slot_c)::value;
        constexpr int ks = decltype(ki)::value;
        typename Engine::packet p;
        const int global_ks = panel * (T::STAGE_K / 128) + ks;
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            const int r = (wm * T::E_M + m) * 16 + lane % 16;
            static_for<2>([&](auto hi) {
                constexpr int h = decltype(hi)::value;
                const int kc = ((ks * 8 + lane / 16 + h * 4) ^ (r % (T::STAGE_K / 16))) * 16;
                set_slice(p.a[m], load<16>(sm, slot * T::PANEL_BYTES + r * T::STAGE_K + kc),
                          number<h * 16>{}, number<(h + 1) * 16>{});
            });
            if constexpr(T::SCALES_PREPACKED) {
                if constexpr(m % 4 == 0) {
                    auto words = make_smem(reinterpret_cast<unsigned*>(arena));
                    const int offset = T::SCALE_A_OFFSET + global_ks * T::SCALE_M_BYTES +
                        (wm * T::M_WORDS + m / 4) * 64 + (lane % 16) * 4;
                    p.sa[m / 4] = load<1>(words, offset / 4)[0];
                }
            } else {
                p.sa[m] = load<1>(ss, T::SCALE_A_OFFSET + global_ks * T::B_M + r)[0];
            }
        });
        static_for<T::E_N>([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            const int offset = slot * T::PANEL_BYTES + T::A_BYTES +
                (n * T::T_N + wn) * 16 * T::STAGE_K + ks * 2048 + lane * 16;
            if constexpr(T::REGISTER_B) p.b[n] = b_queue[slot][ks][n];
            else {
                set_slice(p.b[n], load<16>(sm, offset), 0_I, 16_I);
                set_slice(p.b[n], load<16>(sm, offset + 1024), 16_I, 32_I);
            }
        });
        static_for<T::N_SCALES>([&](auto ni) {
            constexpr int ns = decltype(ni)::value;
            if constexpr(T::SCALES_PREPACKED) {
                if constexpr(ns == 0) {
                    auto words = make_smem(reinterpret_cast<unsigned*>(arena));
                    p.sb[0] = load<1>(words, sb_offset / 4 + global_ks)[0];
                }
            } else {
                p.sb[ns] = load<1>(ss, sb_offset + (ns * sc + global_ks) * 4)[0];
            }
        });
        return p;
    };
    typename Engine::accumulator c{};
    const int panels = (args.k + T::STAGE_K - 1) / T::STAGE_K;
    issue(0_I, 0);
    if (panels > 1) issue(1_I, 1);
    #pragma clang loop unroll(disable)
    for (int base = 0; base < panels; base += 2) {
        static_for<2>([&](auto slot) {
            const int panel = base + decltype(slot)::value;
            if (panel < panels) {
                if (panel + 1 < panels) __builtin_amdgcn_wait_asyncmark(1);
                else __builtin_amdgcn_wait_asyncmark(0);
                __builtin_amdgcn_s_barrier();
                static_for<T::STAGE_K / 128>([&](auto ki) {
                    if (panel * (T::STAGE_K / 128) + decltype(ki)::value < sc) {
                        auto p = read(slot, panel, ki);
                        Engine::contract(p, c);
                    }
                });
                // All consumers release this slot before any wave refills it.
                __builtin_amdgcn_s_barrier();
                if (panel + 2 < panels) issue(slot, panel + 2);
            }
        });
    }
    auto output = make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c) + int64_t(row) * args.stride_c + col,
                            ((rows - 1) * args.stride_c + T::B_N) * sizeof(bf16_t));
    static_for<T::E_M>([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        static_for<T::E_N>([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            const int r = (wm * T::E_M + m) * 16 + lane % 16;
            const int nc = (n * T::T_N + wn) * 16 + (lane / 16) * 4;
            store<4>(output, cast<bf16_t>(c[m * T::E_N + n]), r * args.stride_c + nc);
        });
    });
}
} // namespace opus_gemm_fragment_detail
#endif

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
namespace opus_gemm_fragment_detail {
template<class T>
__device__ auto scheduled_contract(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args,
    int row, int col) {
    using Engine = fragment_engine<T>;
    const int tid = thread_id_x(), lane = tid % 64;
    __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wm = wave / T::T_N, wn = wave % T::T_N;
    const int rows = args.m - row < T::B_M ? args.m - row : T::B_M;
    auto ga = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a) + int64_t(row) * T::FIXED_K,
                       rows * T::FIXED_K);
    auto gb = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) + int64_t(col) * T::FIXED_K,
                       T::B_N * T::FIXED_K);
    auto gsa = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa), args.m * (T::FIXED_K / 128));
    auto gsb = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb), (args.n / 128) * (T::FIXED_K / 128));
    array<array<unsigned, T::E_M>, T::PANELS> sa;
    array<array<unsigned, T::N_SCALES>, T::PANELS> sb;
    if constexpr(T::PRELOAD_SCALES) {
        static_for<T::PANELS>([&](auto ki) {
            constexpr int panel = decltype(ki)::value;
            const int kblock = panel;
            static_for<T::E_M>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int r = row + (wm * T::E_M + m) * 16 + lane % 16;
                sa[panel][m] = load<1>(gsa, kblock * args.stride_sfa + (r < args.m ? r : row))[0];
            });
            static_for<T::N_SCALES>([&](auto si) {
                sb[panel][si] = load<1>(gsb, (col / 128 + decltype(si)::value) * (T::FIXED_K / 128) + kblock)[0];
            });
        });
        __builtin_amdgcn_sched_barrier(0);
    }
    auto fetch = [&](int panel) {
        return Engine::template fetch<!T::PRELOAD_SCALES>(ga, gb, gsa, gsb, panel,
            T::FIXED_K, T::FIXED_K, args.stride_sfa, T::FIXED_K / 128,
            row, col, args.m, lane, wm, wn);
    };
    typename Engine::accumulator c{};
    array<typename Engine::packet, T::DEPTH> queue;
    static_for<(T::PANELS < T::DEPTH ? T::PANELS : T::DEPTH)>([&](auto i) {
        queue[i] = fetch(decltype(i)::value);
    });
    __builtin_amdgcn_sched_barrier(0);
    static_for<T::PANELS>([&](auto i) {
        constexpr int panel = decltype(i)::value, slot = panel % T::DEPTH;
        if constexpr(T::PRELOAD_SCALES) {
            queue[slot].sa = sa[panel];
            queue[slot].sb = sb[panel];
        }
        Engine::contract(queue[slot], c);
        __builtin_amdgcn_sched_barrier(0);
        if constexpr(panel + T::DEPTH < T::PANELS)
            queue[slot] = fetch(panel + T::DEPTH);
        __builtin_amdgcn_sched_barrier(0);
    });
    return c;
}

template<class T, class C>
__device__ void scheduled_store(C& c, bf16_t* destination, int row, int col,
                               int shape_m, int stride_c) {
    const int tid = thread_id_x(), lane = tid % 64;
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wm = wave / T::T_N, wn = wave % T::T_N;
    const int rows = shape_m - row < T::B_M ? shape_m - row : T::B_M;
    auto output = make_gmem(destination + int64_t(row) * stride_c + col,
                            ((rows - 1) * stride_c + T::B_N) * sizeof(bf16_t));
    static_for<T::E_M>([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        static_for<T::E_N>([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            const int r = (wm * T::E_M + m) * 16 + lane % 16;
            const int nc = (n * T::T_N + wn) * 16 + (lane / 16) * 4;
            store<4>(output, cast<bf16_t>(c[m * T::E_N + n]), r * stride_c + nc,
                     0, number<2>{});
        });
    });
}
} // namespace opus_gemm_fragment_detail
#endif

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_scheduled_fragment_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    using namespace opus;
    using T = Traits;
    const int row = block_id_y() * T::B_M, col = block_id_x() * T::B_N;
    auto c = opus_gemm_fragment_detail::scheduled_contract<T>(args, row, col);
    opus_gemm_fragment_detail::scheduled_store<T>(c, reinterpret_cast<bf16_t*>(args.ptr_c),
                                                 row, col, args.m, args.stride_c);
#endif
}

template<class Traits>
__global__ __launch_bounds__(Traits::BLOCK_SIZE, Traits::WG_PER_CU)
void opus_gemm_fragment_kernel(opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    using namespace opus;
    using T = Traits;
    if constexpr (T::FIXED_K > 0) {
        args.k = T::FIXED_K;
        args.stride_a = T::FIXED_K;
        args.stride_b = T::FIXED_K;
        args.stride_sfb = T::FIXED_K / 128;
    }
    if constexpr (T::STAGE_K > 0) {
        opus_gemm_fragment_detail::shared_body<T>(args);
    } else {
    using Engine = opus_gemm_fragment_detail::fragment_engine<T>;
    const int row = block_id_y() * T::B_M, col = block_id_x() * T::B_N;
    const int tid = thread_id_x(), lane = tid % 64;
    __builtin_assume(static_cast<unsigned>(tid) < T::BLOCK_SIZE);
    const int wave = __builtin_amdgcn_readfirstlane(tid / 64);
    const int wm = wave / T::T_N, wn = wave % T::T_N;
    const int rows = args.m - row < T::B_M ? args.m - row : T::B_M;
    auto ga = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a) + int64_t(row) * args.stride_a,
                        rows * args.stride_a);
    auto gb = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) + int64_t(col) * args.stride_b,
                        T::B_N * args.stride_b);
    auto gsa = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa), args.m * (args.k / 128));
    auto gsb = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb), (args.n / 128) * (args.k / 128));
    auto output = make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c) + int64_t(row) * args.stride_c + col,
                            ((rows - 1) * args.stride_c + T::B_N) * sizeof(bf16_t));
    typename Engine::accumulator c{};
    array<typename Engine::packet, T::DEPTH> queue;
    const int panels = args.k / 128;
    auto fetch = [&](int kblock) {
        return Engine::fetch(ga, gb, gsa, gsb, kblock, args.stride_a, args.stride_b,
                             args.stride_sfa, args.stride_sfb, row, col, args.m, lane, wm, wn);
    };
    static_for<T::DEPTH>([&](auto i) {
        if (decltype(i)::value < panels) queue[i] = fetch(decltype(i)::value);
    });
    // Each slot is retired before being filled again. The other slots retain
    // outstanding global loads; scale packing happens only on consumption.
    // Static slot ownership avoids register-array indexing in the K loop.
    #pragma clang loop unroll(disable)
    for (int base = 0; base < panels; base += T::DEPTH) {
        static_for<T::DEPTH>([&](auto i) {
            constexpr int slot = decltype(i)::value;
            if (base + slot < panels) {
                Engine::contract(queue[slot], c);
                __builtin_amdgcn_sched_barrier(0);
                if (base + slot + T::DEPTH < panels)
                    queue[slot] = fetch(base + slot + T::DEPTH);
                __builtin_amdgcn_sched_barrier(0);
            }
        });
    }
    static_for<T::E_M>([&](auto mi) {
        constexpr int m = decltype(mi)::value;
        static_for<T::E_N>([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            const int r = (wm * T::E_M + m) * 16 + lane % 16;
            const int nc = (n * T::T_N + wn) * 16 + (lane / 16) * 4;
            store<4>(output, cast<bf16_t>(c[m * T::E_N + n]), r * args.stride_c + nc);
        });
    });
    }
#endif
}
