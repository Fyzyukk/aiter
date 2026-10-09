// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <type_traits>
#include <cstdint>
#include "opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_4wave_128x128_gfx950.cuh"
#include "opus_gemm_traits_a8w8_mxscale_bpreshuffle_small_gfx950.cuh"
#include "opus_gemm_mxscale_bpreshuffle_small_output_gfx950.cuh"

#if !defined(__HIP_DEVICE_COMPILE__)
template<class T>
__global__ void gemm_a8w8_mxfp8_scale_small_lds_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950) {}
#elif defined(__gfx950__)
template<class T>
__global__ __launch_bounds__(T::BLOCK_SIZE, 1)
void gemm_a8w8_mxfp8_scale_small_lds_kernel(opus_gemm_mxscale_bpreshuffle_kargs_gfx950 args) {
    using namespace opus;
    constexpr int SplitK = T::SPLIT_K;
    using D_C = std::conditional_t<SplitK == 1, bf16_t, float>;
    const int split = SplitK > 1 ? block_id_z() : 0;
    const int lane = thread_id_x() % 64;
    const int wave = __builtin_amdgcn_readfirstlane(thread_id_x() / 64);
    const int wm = wave % T::T_M, wn = wave / T::T_M;
    const int row = block_id_y() * T::B_M, col = block_id_x() * T::B_N;
    const int total_loops = (T::FIXED_K ? T::FIXED_K : args.k) / T::B_K;
    const int per_split = total_loops / SplitK, extra = total_loops % SplitK;
    const int loops = per_split + (split < extra);
    const int tile_begin = split * per_split + (split < extra ? split : extra);
    const int active_stages = loops < T::NUM_STAGES ? loops : T::NUM_STAGES;
    const int matrix_bytes = active_stages * (T::A_STAGE + T::B_STAGE);
    const int scale_bytes = T::B_M * loops;
    auto ga = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a) + row * args.stride_a + tile_begin * 128,
                        static_cast<unsigned>((args.m - row) * args.stride_a - tile_begin * 128));
    auto gb = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) + col * args.stride_b + tile_begin * 2048);
    auto gsa = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa) + row + tile_begin * args.stride_sfa,
                         static_cast<unsigned>(args.m * total_loops - row - tile_begin * args.stride_sfa));
    auto gsb = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb) + (col / 128) * args.stride_sfb + tile_begin);
    auto gc = make_gmem(reinterpret_cast<D_C*>(args.ptr_c) + static_cast<int64_t>(split) * args.m * args.stride_c + row * args.stride_c + col,
                        static_cast<unsigned>(((args.m - row) * args.stride_c - col) * sizeof(D_C)));
    extern __shared__ __attribute__((aligned(16))) char lds[];
    auto sa = make_smem(reinterpret_cast<fp8_t*>(lds));
    auto sb = make_smem(reinterpret_cast<fp8_t*>(lds + active_stages * T::A_STAGE));
    auto ssa = make_smem(reinterpret_cast<unsigned char*>(lds + matrix_bytes));
    auto ssb = make_smem(reinterpret_cast<unsigned char*>(lds + matrix_bytes + scale_bytes));
    const int load_lane = T::XOR_LDS ? lane ^ ((wave % 4) * 2) : lane;
    const auto ugb = make_layout_gb_scale<T>(lane, wm, wn, args.stride_b);
    const auto usb = make_layout_sb_scale<T>(wm, wn);
    const auto urb = make_layout_rb_scale<T>(lane, wn);
    const auto g_b_offsets = layout_to_offsets<16>(ugb);
    const auto s_b_offsets = layout_to_offsets<16>(usb);
    const auto r_b_offsets = layout_to_offsets<16>(urb);
    auto lds_offset = [](int offset) {
        if constexpr (T::XOR_LDS) return offset ^ (((offset >> 10) & 3) << 5);
        else return offset;
    };
    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(seq<T::E_M, T::E_N, 1>{},
        seq<T::T_M, T::T_N, 1>{}, seq<16, 16, 128>{}, mfma_adaptor_swap_ab{});
    using Base = typename decltype(mma)::MMA;
    array<typename Base::vtype_c, T::E_M * T::E_N> c{};
    array<array<unsigned, (T::E_M + 3) / 4>, T::NUM_STAGES> qsa;
    array<array<unsigned, T::B_GROUPS>, T::NUM_STAGES> qsb;
    auto prefetch = [&](auto slot, int kt) {
        constexpr int stage = decltype(slot)::value;
        if constexpr (T::REGISTER_SCALES) {
            qsa[stage] = {};
            static_for<T::E_M>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                const int r = (m * T::T_M + wm) * 16 + lane % 16;
                const int offset = row + r < args.m ? kt * args.stride_sfa + r : -1;
                qsa[stage][m / 4] |= static_cast<unsigned>(load<1>(gsa, offset)[0]) << ((m % 4) * 8);
            });
            static_for<T::B_GROUPS>([&](auto ni) {
                constexpr int n = decltype(ni)::value;
                qsb[stage][n] = load<1>(gsb, n * args.stride_sfb + kt)[0];
            });
        }
        if constexpr (T::FINE_M_LOADS) {
            // M48/M80/M112 need a final A copy with fewer active waves.
            static_for<(T::B_M / 8 + T::NUM_WAVES - 1) / T::NUM_WAVES>([&](auto pass) {
                const int group = decltype(pass)::value * T::NUM_WAVES + wave;
                if (group < T::B_M / 8) {
                    const int r = (group / T::T_M) * 8 * T::T_M + (lane / 8) * T::T_M + wm;
                    async_load<16>(ga, sa.ptr + group * (1024 + T::smem_padding) + stage * T::A_STAGE,
                                   r * args.stride_a + (lane % 8) * 16, kt * 128);
                }
            });
        } else {
            const auto uga = make_layout_ga_scale<T>(load_lane, wm, wn, args.stride_a);
            const auto usa = make_layout_sa_scale<T>(wm, wn);
            async_load<16>(ga, sa.ptr, uga, usa + stage * T::A_STAGE, kt * 128);
        }
        if constexpr (T::XOR_LDS) {
            // Each B copy covers a different 1024-byte group, unlike A's per-wave groups.
            static_for<T::B_N * T::B_K / (T::BLOCK_SIZE * 16)>([&](auto i) {
                constexpr int idx = decltype(i)::value;
                const int mask = ((s_b_offsets[idx] >> 10) & 3) << 5;
                async_load<16>(gb, sb.ptr + s_b_offsets[idx] + stage * T::B_STAGE,
                               g_b_offsets[idx] ^ mask, kt * 2048);
            });
        } else {
            async_load<16>(gb, sb.ptr, ugb, usb + stage * T::B_STAGE, kt * 2048);
        }
    };
    constexpr int distance = T::NUM_STAGES - T::CLUSTER;
    const int initial_tiles = loops <= T::NUM_STAGES ? loops : distance;
    auto load_scales = [&] {
        static_for<(T::MAX_SFA_BYTES + T::BLOCK_SIZE * 16 - 1) / (T::BLOCK_SIZE * 16)>([&](auto pass) {
            const int index = (thread_id_x() + decltype(pass)::value * T::BLOCK_SIZE) * 16;
            if (index < scale_bytes) {
                const int r = index % T::B_M, kt = index / T::B_M;
                const auto value = opus_gemm_4wave_128x128_layout::load_sfa_vector<T>(gsa,
                    kt * args.stride_sfa + r, args.m - row - r);
                store<16>(ssa, value, index);
            }
        });
        if (thread_id_x() < T::B_GROUPS * loops) {
            const int index = thread_id_x();
            store<1>(ssb, load<1>(gsb, (index / loops) * args.stride_sfb + index % loops), index);
        }
    };
    if constexpr (!T::REGISTER_SCALES && T::EARLY_SCALE_LOADS) {
        load_scales();
        __builtin_amdgcn_sched_barrier(0);
    }
    static_for<T::NUM_STAGES>([&](auto i) {
        if (decltype(i)::value < initial_tiles) prefetch(i, decltype(i)::value);
    });
    if constexpr (!T::REGISTER_SCALES && !T::EARLY_SCALE_LOADS) load_scales();
    if (loops <= T::NUM_STAGES || (!T::REGISTER_SCALES && !T::EARLY_SCALE_LOADS)) {
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        if (loops <= T::NUM_STAGES) __builtin_amdgcn_s_barrier();
    }
    // Fine-M's last A copy is issued by only a subset of waves. Leave exactly
    // the later tiles issued by this wave in flight, while preserving all barriers.
    auto wait_for_future_tiles = [&](auto future_tiles) {
        constexpr int tiles = decltype(future_tiles)::value;
        constexpr int partial_waves = (T::B_M / 8) % T::NUM_WAVES;
        if constexpr (T::FINE_M_LOADS && !T::REGISTER_SCALES &&
                      T::EARLY_SCALE_LOADS && partial_waves != 0) {
            if (wave < partial_waves)
                s_waitcnt_vmcnt(number<tiles * (T::VMEM_TILE + 1)>{});
            else
                s_waitcnt_vmcnt(number<tiles * T::VMEM_TILE>{});
        } else {
            s_waitcnt_vmcnt(number<tiles * T::VMEM_TILE>{});
        }
    };
    auto step = [&](auto si, int kt, auto ring) {
        constexpr int stage = decltype(si)::value;
        if constexpr (decltype(ring)::value && T::CLUSTER == 1 && !T::READ_ONLY_DRAIN) {
            if (kt + distance <= loops)
                wait_for_future_tiles(number<T::NUM_STAGES - 2>{});
            else s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        if constexpr (decltype(ring)::value && T::PREFETCH_BEFORE_READ) {
            if (kt + distance < loops) prefetch(number<(stage + distance) % T::NUM_STAGES>{}, kt + distance);
        }
        typename decltype(mma)::vtype_a a;
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            const int r = (m * T::T_M + wm) * 16 + lane % 16;
            const int offset = ((r / (8 * T::T_M)) * T::T_M + r % T::T_M) * (1024 + T::smem_padding) +
                               ((r / T::T_M) % 8) * 128 + (lane / 16) * 16 + stage * T::A_STAGE;
            set_slice(a, load<16>(sa, lds_offset(offset)), number<m * 32>{}, number<m * 32 + 16>{});
            set_slice(a, load<16>(sa, lds_offset(offset + 64)), number<m * 32 + 16>{}, number<(m + 1) * 32>{});
        });
        typename decltype(mma)::vtype_b b;
        if constexpr (T::XOR_LDS) {
            static_for<T::E_N * 2>([&](auto i) {
                constexpr int idx = decltype(i)::value;
                set_slice(b, load<16>(sb, lds_offset(r_b_offsets[idx] + stage * T::B_STAGE)),
                          number<idx * 16>{}, number<(idx + 1) * 16>{});
            });
        } else {
            b = load<16>(sb, urb + stage * T::B_STAGE);
        }
        array<unsigned, (T::E_M + 3) / 4> sfa{};
        array<unsigned, T::B_GROUPS> sfb;
        if constexpr (T::REGISTER_SCALES) {
            sfa = qsa[stage];
            sfb = qsb[stage];
        } else {
            static_for<T::E_M>([&](auto mi) {
                constexpr int m = decltype(mi)::value;
                sfa[m / 4] |= static_cast<unsigned>(load<1>(ssa, kt * T::B_M + (m * T::T_M + wm) * 16 + lane % 16)[0]) << ((m % 4) * 8);
            });
            static_for<T::B_GROUPS>([&](auto ni) {
                sfb[decltype(ni)::value] = load<1>(ssb, decltype(ni)::value * loops + kt)[0];
            });
        }
        if constexpr (decltype(ring)::value && !T::PREFETCH_BEFORE_READ) {
            if (kt + distance < loops) prefetch(number<(stage + distance) % T::NUM_STAGES>{}, kt + distance);
        }
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value, ci = m * T::E_N + n;
                constexpr int sg = n / (128 / (T::T_N * 16));
                c[ci] = Base{}(slice(a, number<m * 32>{}, number<(m + 1) * 32>{}),
                               slice(b, number<n * 32>{}, number<(n + 1) * 32>{}), c[ci],
                               static_cast<int>(sfa[m / 4]), static_cast<int>(sfb[sg]), number<m % 4>{}, 0_I);
            });
        });
    };
    if (loops <= T::NUM_STAGES) {
        // Short K: the prologue has already loaded every matrix tile.
        static_for<T::NUM_STAGES>([&](auto i) {
            if (decltype(i)::value < loops) step(i, decltype(i)::value, 0_I);
        });
    } else if constexpr (T::CLUSTER == 1 && T::READ_ONLY_DRAIN) {
        const int drain_begin = loops - distance;
        for (int base = 0; base < drain_begin; base += T::NUM_STAGES) {
            static_for<T::NUM_STAGES>([&](auto i) {
                if (base + decltype(i)::value < drain_begin) {
                    wait_for_future_tiles(number<T::NUM_STAGES - 2>{});
                    s_waitcnt_lgkmcnt(0_I);
                    __builtin_amdgcn_s_barrier();
                    step(i, base + decltype(i)::value, 1_I);
                }
            });
        }
        // No ring slot is overwritten after this wait; the whole drain is read-only.
        s_waitcnt_vmcnt(0_I);
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        for (int base = (drain_begin / T::NUM_STAGES) * T::NUM_STAGES;
             base < loops; base += T::NUM_STAGES) {
            static_for<T::NUM_STAGES>([&](auto i) {
                const int kt = base + decltype(i)::value;
                if (kt >= drain_begin && kt < loops) step(i, kt, 0_I);
            });
        }
    } else if constexpr (T::CLUSTER == 1) {
        for (int base = 0; base < loops; base += T::NUM_STAGES) {
            static_for<T::NUM_STAGES>([&](auto i) {
                if (base + decltype(i)::value < loops)
                    step(i, base + decltype(i)::value, 1_I);
            });
        }
    } else {
        // Each barrier makes a whole K group visible and protects its ring slots
        // from overwrite. The wait count leaves later groups in flight.
        for (int base = 0; base < loops; base += T::NUM_STAGES) {
            static_for<T::NUM_STAGES / T::CLUSTER>([&](auto group) {
                constexpr int stage = decltype(group)::value * T::CLUSTER;
                if (base + stage < loops) {
                    // Only leave a full future group in flight. A partial K
                    // tail may have fewer pending loads than the steady state.
                    if (base + stage + distance <= loops)
                        wait_for_future_tiles(number<T::NUM_STAGES - 2 * T::CLUSTER>{});
                    else s_waitcnt_vmcnt(0_I);
                    s_waitcnt_lgkmcnt(0_I);
                    __builtin_amdgcn_s_barrier();
                    static_for<T::CLUSTER>([&](auto i) {
                        if (base + stage + decltype(i)::value < loops)
                            step(number<stage + decltype(i)::value>{},
                                 base + stage + decltype(i)::value, 1_I);
                    });
                }
            });
        }
    }
    if constexpr (SplitK > 1) {
        const auto pc = opus::make_tuple(wm, lane % mma.grpn_c, wn, lane / mma.grpn_c);
        const auto uc = partition_layout_c<4>(mma, opus::make_tuple(args.stride_c, 1_I), pc);
        const auto offsets = layout_to_offsets<4>(uc);
        static_for<T::E_M * T::E_N>([&](auto i) {
            store<4>(gc, c[decltype(i)::value], offsets[decltype(i)::value], 0, number<T::STORE_CACHE>{});
        });
    } else if constexpr (T::OUTPUT == 2) {
        opus_gemm_small_output::store_mfma16_packed<T>(c, gc, lane, wm, wn, args.stride_c);
    } else if constexpr (T::OUTPUT == 1) {
        // All waves finish reading matrix LDS before it becomes the C buffer.
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        auto sc = make_smem(reinterpret_cast<bf16_t*>(lds));
        constexpr int stride = T::B_N + 8;
        const auto pc = opus::make_tuple(wm, lane % mma.grpn_c, wn, lane / mma.grpn_c);
        const auto uc = partition_layout_c<4>(mma, opus::make_tuple(number<stride>{}, 1_I), pc);
        const auto offsets = layout_to_offsets<4>(uc);
        static_for<T::E_M * T::E_N>([&](auto i) {
            store<4>(sc, cast<bf16_t>(c[decltype(i)::value]), offsets[decltype(i)::value]);
        });
        s_waitcnt_lgkmcnt(0_I);
        __builtin_amdgcn_s_barrier();
        static_for<T::B_M * T::B_N / (T::BLOCK_SIZE * 8)>([&](auto pass) {
            const int index = (thread_id_x() + decltype(pass)::value * T::BLOCK_SIZE) * 8;
            const int r = index / T::B_N, n = index % T::B_N;
            store<8>(gc, load<8>(sc, r * stride + n), r * args.stride_c + n, 0, number<2>{});
        });
    } else {
        const auto pc = opus::make_tuple(wm, lane % mma.grpn_c, wn, lane / mma.grpn_c);
        const auto uc = partition_layout_c<4>(mma, opus::make_tuple(args.stride_c, 1_I), pc);
        const auto offsets = layout_to_offsets<4>(uc);
        static_for<T::E_M * T::E_N>([&](auto i) {
            store<4>(gc, cast<bf16_t>(c[decltype(i)::value]), offsets[decltype(i)::value]);
        });
    }
}
#endif

template<int SplitK, int Vec=4, int Block=128>
__global__ __launch_bounds__(Block, 2)
void opus_gemm_mxscale_bpreshuffle_reduce_kernel(const float* __restrict__ workspace, opus::bf16_t* __restrict__ out, int elements) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    using namespace opus;
    const int index = (block_id_x() * Block + thread_id_x()) * Vec;
    if (index >= elements) return;
    vector_t<float, Vec> acc{};
    static_for<SplitK>([&](auto split) {
        auto source = make_gmem(workspace + static_cast<int64_t>(decltype(split)::value) * elements,
                                static_cast<unsigned>(elements * sizeof(float)));
        static_for<Vec / 4>([&](auto part) {
            constexpr int offset = decltype(part)::value * 4;
            const auto value = load<4>(source, index + offset);
            static_for<4>([&](auto elem) { acc[offset + decltype(elem)::value] += value[decltype(elem)::value]; });
        });
    });
    auto target = make_gmem(out, static_cast<unsigned>(elements * sizeof(bf16_t)));
    constexpr int store_vec = Vec < 8 ? Vec : 8;
    static_for<Vec / store_vec>([&](auto part) {
        constexpr int offset = decltype(part)::value * store_vec;
        store<store_vec>(target, cast<bf16_t>(slice(acc, number<offset>{}, number<offset+store_vec>{})), index+offset);
    });
#endif
}
