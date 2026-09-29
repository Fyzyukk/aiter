// SPDX-License-Identifier: Apache-2.0
#pragma once
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
    const int lane = thread_id_x() % 64;
    const int wave = __builtin_amdgcn_readfirstlane(thread_id_x() / 64);
    const int wm = wave % T::T_M, wn = wave / T::T_M;
    const int row = block_id_y() * T::B_M, col = block_id_x() * T::B_N;
    const int loops = args.k / T::B_K;
    const int active_stages = loops < T::NUM_STAGES ? loops : T::NUM_STAGES;
    const int matrix_bytes = active_stages * (T::A_STAGE + T::B_STAGE);
    const int scale_bytes = T::B_M * loops;
    auto ga = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_a) + row * args.stride_a,
                        static_cast<unsigned>((args.m - row) * args.stride_a));
    auto gb = make_gmem(reinterpret_cast<const fp8_t*>(args.ptr_b) + col * args.stride_b);
    auto gsa = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfa) + row);
    auto gsb = make_gmem(reinterpret_cast<const unsigned char*>(args.ptr_sfb) + (col / 128) * args.stride_sfb);
    auto gc = make_gmem(reinterpret_cast<bf16_t*>(args.ptr_c) + row * args.stride_c + col,
                        static_cast<unsigned>(((args.m - row) * args.stride_c - col) * 2));
    extern __shared__ __attribute__((aligned(16))) char lds[];
    auto sa = make_smem(reinterpret_cast<fp8_t*>(lds));
    auto sb = make_smem(reinterpret_cast<fp8_t*>(lds + active_stages * T::A_STAGE));
    auto ssa = make_smem(reinterpret_cast<unsigned char*>(lds + matrix_bytes));
    auto ssb = make_smem(reinterpret_cast<unsigned char*>(lds + matrix_bytes + scale_bytes));
    const auto uga = make_layout_ga_scale<T>(lane, wm, wn, args.stride_a);
    const auto usa = make_layout_sa_scale<T>(wm, wn);
    const auto ugb = make_layout_gb_scale<T>(lane, wm, wn, args.stride_b);
    const auto usb = make_layout_sb_scale<T>(wm, wn);
    const auto urb = make_layout_rb_scale<T>(lane, wn);
    auto mma = make_tiled_mma<fp8_t, fp8_t, fp32_t>(seq<T::E_M, T::E_N, 1>{},
        seq<T::T_M, T::T_N, 1>{}, seq<16, 16, 128>{}, mfma_adaptor_swap_ab{});
    using Base = typename decltype(mma)::MMA;
    array<typename Base::vtype_c, T::E_M * T::E_N> c{};
    auto prefetch = [&](int stage, int kt) {
        async_load<16>(ga, sa.ptr, uga, usa + stage * T::A_STAGE, kt * 128);
        async_load<16>(gb, sb.ptr, ugb, usb + stage * T::B_STAGE, kt * 2048);
    };
    constexpr int distance = T::NUM_STAGES - T::CLUSTER;
    const int initial_tiles = loops <= T::NUM_STAGES ? loops : distance;
    static_for<T::NUM_STAGES>([&](auto i) {
        if (decltype(i)::value < initial_tiles) prefetch(decltype(i)::value, decltype(i)::value);
    });
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
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();
    auto step = [&](auto si, int kt, auto ring) {
        constexpr int stage = decltype(si)::value;
        if constexpr (decltype(ring)::value && T::CLUSTER == 1) {
            if (kt + distance <= loops)
                s_waitcnt_vmcnt(number<(T::NUM_STAGES - 2) * T::VMEM_TILE>{});
            else s_waitcnt_vmcnt(0_I);
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        typename decltype(mma)::vtype_a a;
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            const int r = (m * T::T_M + wm) * 16 + lane % 16;
            const int offset = ((r / (8 * T::T_M)) * T::T_M + r % T::T_M) * 1056 +
                               ((r / T::T_M) % 8) * 128 + (lane / 16) * 16 + stage * T::A_STAGE;
            set_slice(a, load<16>(sa, offset), number<m * 32>{}, number<m * 32 + 16>{});
            set_slice(a, load<16>(sa, offset + 64), number<m * 32 + 16>{}, number<(m + 1) * 32>{});
        });
        const auto b = load<16>(sb, urb + stage * T::B_STAGE);
        unsigned sfa = 0;
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            sfa |= static_cast<unsigned>(load<1>(ssa, kt * T::B_M + (m * T::T_M + wm) * 16 + lane % 16)[0]) << (m * 8);
        });
        array<unsigned, T::B_GROUPS> sfb;
        static_for<T::B_GROUPS>([&](auto ni) {
            sfb[decltype(ni)::value] = load<1>(ssb, decltype(ni)::value * loops + kt)[0];
        });
        if constexpr (decltype(ring)::value) {
            if (kt + distance < loops) prefetch((stage + distance) % T::NUM_STAGES, kt + distance);
        }
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value, ci = m * T::E_N + n;
                constexpr int sg = n / (128 / (T::T_N * 16));
                c[ci] = Base{}(slice(a, number<m * 32>{}, number<(m + 1) * 32>{}),
                               slice(b, number<n * 32>{}, number<(n + 1) * 32>{}), c[ci],
                               static_cast<int>(sfa), static_cast<int>(sfb[sg]), number<m>{}, 0_I);
            });
        });
    };
    if (loops <= T::NUM_STAGES) {
        // Short K: the prologue has already loaded every matrix tile.
        static_for<T::NUM_STAGES>([&](auto i) {
            if (decltype(i)::value < loops) step(i, decltype(i)::value, 0_I);
        });
    } else if constexpr (T::CLUSTER == 1) {
        // Keep the wait in step so each ring slot can be expanded independently.
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
                        s_waitcnt_vmcnt(number<(T::NUM_STAGES - 2 * T::CLUSTER) * T::VMEM_TILE>{});
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
    if constexpr (T::OUTPUT == 2) {
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
