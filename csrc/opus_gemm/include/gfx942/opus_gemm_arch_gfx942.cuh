// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx942 a16w16 selector and strict kid dispatch.
#pragma once

#include "../opus_gemm_arch.cuh"
#include "../opus_gemm_common.cuh"
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_lookup.h"
#include "opus_gemm_a16w16_tune_lookup.h"
#include "opus_gemm_a8w8_tune_lookup.h"
#include "opus_gemm_manifest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

#ifndef OPUS_A16W16_DISPATCH_KERNEL_TYPES_DEFINED
#define OPUS_A16W16_DISPATCH_KERNEL_TYPES_DEFINED
using OpusA16W16Kernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    std::optional<aiter_tensor_t>, int);
using OpusA16W16WorkspaceKernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    aiter_tensor_t&, std::optional<aiter_tensor_t>, int);
#endif

namespace opus_gfx942_detail
{
struct OpusA16W16Shape
{
    int M;
    int N;
    int K;
};

struct OpusA16W16RuntimeEntry
{
    OpusA16W16Shape key;
    int kid;
};

struct OpusA16W16TuneEntry
{
    int kid;
    OpusA16W16Kernel func;
};

struct OpusA16W16WorkspaceTuneEntry
{
    int kid;
    OpusA16W16WorkspaceKernel func;
};

template <typename Entry, size_t Size>
inline const Entry* find_kid(const std::array<Entry, Size>& entries, int kid)
{
    const auto it = std::lower_bound(
        entries.begin(), entries.end(), kid,
        [](const Entry& entry, int value) { return entry.kid < value; });
    return it != entries.end() && it->kid == kid ? &*it : nullptr;
}

template <size_t Size>
inline int find_shape_kid(const std::array<OpusA16W16RuntimeEntry, Size>& entries,
                          int M,
                          int N,
                          int K)
{
    const OpusA16W16Shape key{M, N, K};
    const auto it = std::lower_bound(
        entries.begin(), entries.end(), key,
        [](const OpusA16W16RuntimeEntry& entry, const OpusA16W16Shape& value) {
            if(entry.key.M != value.M) return entry.key.M < value.M;
            if(entry.key.N != value.N) return entry.key.N < value.N;
            return entry.key.K < value.K;
        });
    if(it == entries.end() || it->key.M != M || it->key.N != N || it->key.K != K)
        return -1;
    return it->kid;
}

inline const OpusA16W16WorkspaceTuneEntry* workspace_entry(int kid)
{
    static constexpr std::array<
        OpusA16W16WorkspaceTuneEntry,
        GENERATE_A16W16_WORKSPACE_TUNE_LOOKUP_GFX942_SIZE>
        kWorkspace = {{GENERATE_A16W16_WORKSPACE_TUNE_LOOKUP_GFX942}};
    return find_kid(kWorkspace, kid);
}

// Pure integer form of the legacy gfx942 heuristic.  Keeping this probe free
// of launcher symbols is required now that workspace launchers have a distinct
// function type.  The returned kids are part of HEURISTIC_DEFAULT_KIDS_GFX942.
inline bool split_barrier_ok(int N, int K)
{
    const int loops = (K + 63) / 64;
    return N % 16 == 0 && K % 64 == 0 && loops >= 2 && loops % 2 == 0;
}

inline bool bf16ws_band(int M, int N, int K)
{
    return K >= 4096 && K % 64 == 0 && M >= 104 && M <= 608 &&
           (N == 256 || (N >= 512 && N <= 2048));
}

inline int heuristic_bf16_kid(int M, int N, int K)
{
    const bool k64_ok = K % 64 == 0;
    const bool k32_ok = K % 32 == 0;
    const bool wkc_bk64_ok = K >= 4096 && K % 512 == 0;
    const bool p1_ok = K % 128 == 0;

    if(K == 4096)
    {
        if(p1_ok && (M == 48 || M == 64) && N == 1024) return 10213;
        if(p1_ok && ((M == 128 && N == 512) || (M == 256 && N == 256)))
            return 10213;
        if(p1_ok && M == 512 && N == 256) return 10203;
        if((M == 48 || M == 64) && N >= 1536 && N <= 2048) return 10205;
        if((M == 128 && N == 1024) || (M == 256 && N == 512)) return 10205;
        if((M == 128 && N >= 1536 && N <= 2048) ||
           (M == 256 && N == 1024) || (M == 512 && N == 512))
            return 10200;
    }

    if(K >= 1024 && k32_ok && N >= 1536 && M <= 32)
    {
        if(M <= 4 && N >= 4096) return 10300;
        if(M <= 16) return wkc_bk64_ok ? 10305 : 10301;
        return M == 32 && K == 4096 && wkc_bk64_ok ? 10305 : 10303;
    }

    if(K >= 512 && k64_ok &&
       (N <= 64 || (M <= 128 && N <= 1024) || (M <= 8 && N <= 1536)))
    {
        if(N <= 64 && M > 128) return 10302;
        if(N <= 256 || M <= 8 || (M <= 16 && N <= 800)) return 10300;
        return 10302;
    }

    if(bf16ws_band(M, N, K)) return 10210;

    if(N == 384 && K >= 4096)
    {
        if(M <= 128) return 10302;
        if(M <= 224) return 10201;
        if(M >= 392 && M <= 512) return 10204;
        return 10200;
    }

    if(k64_ok && N >= 4096 && K <= 3200)
        return K <= 640 && M <= 128 ? 10001 : 10000;

    if(split_barrier_ok(N, K) && M >= 128) return 10000;
    if(N <= 256 && p1_ok) return 10201;
    return 10200;
}

inline int heuristic_non_bf16_or_bias_kid(int N, int K)
{
    return N <= 256 && K % 128 == 0 ? 10201 : 10200;
}

using OpusA8W8BlockscaleBPreshuffleKernel = void (*)(
    aiter_tensor_t&, aiter_tensor_t&, aiter_tensor_t&,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);

struct OpusA8W8TuneEntry
{
    int kid;
    OpusA8W8BlockscaleBPreshuffleKernel func;
};
} // namespace opus_gfx942_detail

template <typename CDataType>
inline OpusA16W16Kernel opus_a16w16_tune_dispatch_gfx942(int id);

template <>
inline OpusA16W16Kernel opus_a16w16_tune_dispatch_gfx942<bf16_t>(int id)
{
    using namespace opus_gfx942_detail;
    static constexpr std::array<
        OpusA16W16TuneEntry, GENERATE_A16W16_TUNE_LOOKUP_GFX942_BF16_SIZE>
        kTune = {{GENERATE_A16W16_TUNE_LOOKUP_GFX942_BF16(bf16_t)}};
    const auto* entry = find_kid(kTune, id);
    AITER_CHECK(entry != nullptr,
                "Kernel id ",
                id,
                " not found in gfx942 a16w16 bf16 non-workspace table");
    return entry->func;
}

template <>
inline OpusA16W16Kernel opus_a16w16_tune_dispatch_gfx942<fp32_t>(int id)
{
    using namespace opus_gfx942_detail;
    static constexpr std::array<
        OpusA16W16TuneEntry, GENERATE_A16W16_TUNE_LOOKUP_GFX942_FP32_SIZE>
        kTune = {{GENERATE_A16W16_TUNE_LOOKUP_GFX942_FP32(fp32_t)}};
    const auto* entry = find_kid(kTune, id);
    AITER_CHECK(entry != nullptr,
                "Kernel id ",
                id,
                " not found in gfx942 a16w16 fp32 non-workspace table");
    return entry->func;
}

inline bool opus_a16w16_has_workspace_kernel_gfx942(int id)
{
    return opus_gfx942_detail::workspace_entry(id) != nullptr;
}

inline OpusA16W16WorkspaceKernel
opus_a16w16_workspace_dispatch_gfx942(int id)
{
    const auto* entry = opus_gfx942_detail::workspace_entry(id);
    AITER_CHECK(entry != nullptr,
                "Kernel id ",
                id,
                " not found in gfx942 a16w16 workspace table");
    return entry->func;
}

template <typename CDataType>
inline int opus_select_a16w16_kid_gfx942(
    int M, int N, int K, int batch, bool has_bias = false);

template <>
inline int opus_select_a16w16_kid_gfx942<bf16_t>(
    int M, int N, int K, int batch, bool has_bias)
{
    using namespace opus_gfx942_detail;
    static constexpr std::array<
        OpusA16W16RuntimeEntry, GENERATE_OPUS_LOOKUP_TABLE_GFX942_BF16_SIZE>
        kLookup = {{GENERATE_OPUS_LOOKUP_TABLE_GFX942_BF16}};
    const int tuned_kid = find_shape_kid(kLookup, M, N, K);
    if(tuned_kid >= 0) return tuned_kid;
    (void)batch;
    return has_bias ? heuristic_non_bf16_or_bias_kid(N, K)
                    : heuristic_bf16_kid(M, N, K);
}

template <>
inline int opus_select_a16w16_kid_gfx942<fp32_t>(
    int M, int N, int K, int batch, bool has_bias)
{
    using namespace opus_gfx942_detail;
    static constexpr std::array<
        OpusA16W16RuntimeEntry, GENERATE_OPUS_LOOKUP_TABLE_GFX942_FP32_SIZE>
        kLookup = {{GENERATE_OPUS_LOOKUP_TABLE_GFX942_FP32}};
    const int tuned_kid = find_shape_kid(kLookup, M, N, K);
    if(tuned_kid >= 0) return tuned_kid;
    (void)M;
    (void)batch;
    (void)has_bias;
    return heuristic_non_bf16_or_bias_kid(N, K);
}

// A8W8 uses a separate unchanged five-argument scale ABI and table.
inline opus_gfx942_detail::OpusA8W8BlockscaleBPreshuffleKernel
opus_a8w8_tune_dispatch_gfx942(int id)
{
    using namespace opus_gfx942_detail;
    static constexpr OpusA8W8TuneEntry kTune[] = {
        GENERATE_A8W8_TUNE_LOOKUP_BF16(bf16_t)
    };
    constexpr size_t kSize = sizeof(kTune) / sizeof(kTune[0]);
    const auto it = std::lower_bound(
        kTune, kTune + kSize, id,
        [](const OpusA8W8TuneEntry& entry, int value) { return entry.kid < value; });
    AITER_CHECK(it != kTune + kSize && it->kid == id,
                "Kernel id ",
                id,
                " not found in gfx942 a8w8 bf16 tune table");
    return it->func;
}
