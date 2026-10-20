// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "opus_gemm_traits_a8w8_blockscale_mxscale_bpreshuffle_lds_ring_gfx950.cuh"

template <class T>
__global__
__launch_bounds__(T::BLOCK_SIZE, T::WG_PER_CU) void opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_kernel(
    opus_gemm_blockscale_mxscale_bpreshuffle_kargs_gfx950 args) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    using namespace opus;
    using D_C = std::conditional_t<T::SPLIT_K == 1, bf16_t, float>;
    const int row = block_id_y() * T::B_M, col = block_id_x() * T::B_N;
    const int split = block_id_z(), lane = thread_id_x() % 64;
    const int wave = __builtin_amdgcn_readfirstlane(thread_id_x() / 64);
    const int wm = wave % T::T_M, wn = wave / T::T_M;
    const int total_loops = (T::FIXED_K ? T::FIXED_K : args.k) / T::B_K;
    const int per_split = total_loops / T::SPLIT_K, extra = total_loops % T::SPLIT_K;
    const int loops = per_split + (split < extra);
    const int tile_begin = split * per_split + (split < extra ? split : extra);
    const int active_stages = loops < T::NUM_STAGES ? loops : T::NUM_STAGES;
    const int matrix_bytes = active_stages * (T::A_STAGE + T::B_STAGE);
    const int scale_bytes = T::B_M * loops;

    // Prologue: tile views, LDS ring, scale panels and accumulator layout.
    auto ga = make_gmem(reinterpret_cast<const fp8_t *>(args.ptr_a) + row * args.stride_a +
                            tile_begin * 128,
                        static_cast<unsigned>((args.m - row) * args.stride_a - tile_begin * 128));
    auto gb = make_gmem(reinterpret_cast<const fp8_t *>(args.ptr_b) + col * args.stride_b +
                        tile_begin * 2048);
    auto gsa = make_gmem(
        reinterpret_cast<const unsigned char *>(args.ptr_sfa) + row + tile_begin * args.stride_sfa,
        static_cast<unsigned>((total_loops - tile_begin - 1) * args.stride_sfa + args.m - row));
    auto gsb = make_gmem(reinterpret_cast<const unsigned char *>(args.ptr_sfb) +
                         (col / 128) * args.stride_sfb + tile_begin);
    auto gc = make_gmem(
        reinterpret_cast<D_C *>(args.ptr_c) + static_cast<int64_t>(split) * args.m * args.stride_c +
            static_cast<int64_t>(row) * args.stride_c + col,
        static_cast<unsigned>(
            (((args.m - row < T::B_M ? args.m - row : T::B_M) - 1) * args.stride_c + T::B_N) *
            sizeof(D_C)));
    extern __shared__ __attribute__((aligned(16))) char lds[];
    auto sa = make_smem(reinterpret_cast<fp8_t *>(lds));
    auto sb = make_smem(reinterpret_cast<fp8_t *>(lds + active_stages * T::A_STAGE));
    auto ssa = make_smem(reinterpret_cast<unsigned char *>(lds + matrix_bytes));
    auto ssb = make_smem(reinterpret_cast<unsigned char *>(lds + matrix_bytes + scale_bytes));
    const auto ugb = opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_gb_scale<T>(
        lane, wm, wn, args.stride_b);
    const auto usb = opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_sb_scale<T>(wm, wn);
    const auto urb =
        opus_gemm_blockscale_mxscale_bpreshuffle_tiled_make_layout_rb_scale<T>(lane, wn);
    auto mma =
        make_tiled_mma<fp8_t, fp8_t, fp32_t>(seq<T::E_M, T::E_N, 1>{}, seq<T::T_M, T::T_N, 1>{},
                                             seq<16, 16, 128>{}, mfma_adaptor_swap_ab{});
    using Base = typename decltype(mma)::MMA;
    array<typename Base::vtype_c, T::E_M * T::E_N> c{};
    auto prefetch = [&](auto slot, int kt) {
        constexpr int stage = decltype(slot)::value;
        static_for<(T::B_M / 8 + T::NUM_WAVES - 1) / T::NUM_WAVES>([&](auto pass) {
            const int group = decltype(pass)::value * T::NUM_WAVES + wave;
            if (group < T::B_M / 8) {
                const int r = (group / T::T_M) * 8 * T::T_M + (lane / 8) * T::T_M + wm;
                async_load<16>(ga, sa.ptr + group * (1024 + T::smem_padding) + stage * T::A_STAGE,
                               r * args.stride_a + (lane % 8) * 16, kt * 128, 0_I,
                               number<T::CACHECTL_A>{});
            }
        });
        async_load<16>(gb, sb.ptr, ugb, usb + stage * T::B_STAGE, kt * 2048, 0_I,
                       number<T::CACHECTL_B>{});
    };
    static_for<(T::MAX_SFA_BYTES + T::BLOCK_SIZE * 16 - 1) / (T::BLOCK_SIZE * 16)>([&](auto pass) {
        const int index = (thread_id_x() + decltype(pass)::value * T::BLOCK_SIZE) * 16;
        if (index < scale_bytes) {
            const int r = index % T::B_M, kt = index / T::B_M;
            const auto value =
                opus_gemm_blockscale_mxscale_bpreshuffle_panel_half_panel::load_sfa_vector<T>(
                    gsa, kt * args.stride_sfa + r, args.m - row - r);
            store<16>(ssa, value, index);
        }
    });
    if (thread_id_x() < T::B_GROUPS * loops) {
        const int index = thread_id_x();
        store<1>(ssb, load<1>(gsb, (index / loops) * args.stride_sfb + index % loops), index);
    }
    constexpr int distance = T::NUM_STAGES - 1;
    const int initial_tiles = loops <= T::NUM_STAGES ? loops : distance;
    static_for<T::NUM_STAGES>([&](auto i) {
        if (decltype(i)::value < initial_tiles)
            prefetch(i, decltype(i)::value);
    });
    // Publish all scale bytes before any wave reads them. The short path has no refills.
    s_waitcnt_vmcnt(0_I);
    s_waitcnt_lgkmcnt(0_I);
    __builtin_amdgcn_s_barrier();

    // Main loop: wait for a ring slot, read operands, refill the released slot, and issue MFMA.
    auto step = [&](auto si, int kt, auto refill) {
        constexpr int stage = decltype(si)::value;
        if constexpr (decltype(refill)::value) {
            if (kt + distance <= loops) {
                constexpr int pending = (T::NUM_STAGES - 2) * T::VMEM_TILE;
                s_waitcnt_vmcnt(number<(pending < 63 ? pending : 63)>{});
            } else {
                s_waitcnt_vmcnt(0_I);
            }
            s_waitcnt_lgkmcnt(0_I);
            __builtin_amdgcn_s_barrier();
        }
        typename decltype(mma)::vtype_a a;
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            const int r = (m * T::T_M + wm) * 16 + lane % 16;
            const int offset = (r / (8 * T::T_M)) * T::T_M * (1024 + T::smem_padding) +
                               (r % T::T_M) * (1024 + T::smem_padding) + ((r / T::T_M) % 8) * 128 +
                               (lane / 16) * 16 + stage * T::A_STAGE;
            set_slice(a, load<16>(sa, offset), number<m * 32>{}, number<m * 32 + 16>{});
            set_slice(a, load<16>(sa, offset + 64), number<m * 32 + 16>{}, number<(m + 1) * 32>{});
        });
        const auto b = load<16>(sb, urb + stage * T::B_STAGE);
        array<unsigned, (T::E_M + 3) / 4> sfa{};
        array<unsigned, T::B_GROUPS> sfb;
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            sfa[m / 4] |= static_cast<unsigned>(
                              load<1>(ssa, kt * T::B_M + (m * T::T_M + wm) * 16 + lane % 16)[0])
                          << ((m % 4) * 8);
        });
        static_for<T::B_GROUPS>([&](auto ni) {
            constexpr int n = decltype(ni)::value;
            sfb[n] = load<1>(ssb, n * loops + kt)[0];
        });
        if constexpr (decltype(refill)::value) {
            if (kt + distance < loops)
                prefetch(number<(stage + distance) % T::NUM_STAGES>{}, kt + distance);
        }
        static_for<T::E_M>([&](auto mi) {
            constexpr int m = decltype(mi)::value;
            static_for<T::E_N>([&](auto ni) {
                constexpr int n = decltype(ni)::value, ci = m * T::E_N + n;
                constexpr int sg = n / (128 / (T::T_N * 16));
                c[ci] = Base{}(slice(a, number<m * 32>{}, number<(m + 1) * 32>{}),
                               slice(b, number<n * 32>{}, number<(n + 1) * 32>{}), c[ci],
                               static_cast<int>(sfa[m / 4]), static_cast<int>(sfb[sg]),
                               number<m % 4>{}, 0_I);
            });
        });
    };
    if (loops <= T::NUM_STAGES) {
        static_for<T::NUM_STAGES>([&](auto i) {
            if (decltype(i)::value < loops)
                step(i, decltype(i)::value, 0_I);
        });
    } else {
#pragma clang loop unroll(disable)
        for (int base = 0; base < loops; base += T::NUM_STAGES) {
            static_for<T::NUM_STAGES>([&](auto i) {
                if (base + decltype(i)::value < loops)
                    step(i, base + decltype(i)::value, 1_I);
            });
        }
    }
    // Epilogue: complete FP32 partials for split-K, otherwise convert to BF16.
    if constexpr (T::SPLIT_K == 1 && T::OUTPUT == 2) {
        opus_gemm_blockscale_mxscale_bpreshuffle_register_prefetch_mfma16_output::
            store_mfma16_packed<T>(c, gc, lane, wm, wn, args.stride_c);
    } else {
        const auto pc = opus::make_tuple(wm, lane % mma.grpn_c, wn, lane / mma.grpn_c);
        const auto uc = partition_layout_c<4>(mma, opus::make_tuple(args.stride_c, 1_I), pc);
        const auto offsets = layout_to_offsets<4>(uc);
        static_for<T::E_M * T::E_N>([&](auto i) {
            store<4>(gc, cast<D_C>(c[decltype(i)::value]), offsets[decltype(i)::value], 0,
                     number<T::STORE_CACHE>{});
        });
    }
#endif
}

template <int SplitK, int Vec = 4, int Block = 128>
__global__
__launch_bounds__(Block, 2) void opus_gemm_blockscale_mxscale_bpreshuffle_lds_ring_reduce(
    const float *__restrict__ workspace, opus::bf16_t *__restrict__ out, int elements) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx950__)
    using namespace opus;
    const int index = (block_id_x() * Block + thread_id_x()) * Vec;
    if (index >= elements)
        return;
    vector_t<float, Vec> acc{};
    static_for<SplitK>([&](auto split) {
        auto source = make_gmem(workspace + static_cast<int64_t>(decltype(split)::value) * elements,
                                static_cast<unsigned>(elements * sizeof(float)));
        static_for<Vec / 4>([&](auto part) {
            constexpr int offset = decltype(part)::value * 4;
            const auto value = load<4>(source, index + offset);
            static_for<4>([&](auto elem) {
                acc[offset + decltype(elem)::value] += value[decltype(elem)::value];
            });
        });
    });
    auto target = make_gmem(out, static_cast<unsigned>(elements * sizeof(bf16_t)));
    constexpr int store_vec = Vec < 8 ? Vec : 8;
    static_for<Vec / store_vec>([&](auto part) {
        constexpr int offset = decltype(part)::value * store_vec;
        store<store_vec>(target,
                         cast<bf16_t>(slice(acc, number<offset>{}, number<offset + store_vec>{})),
                         index + offset);
    });
#endif
}
