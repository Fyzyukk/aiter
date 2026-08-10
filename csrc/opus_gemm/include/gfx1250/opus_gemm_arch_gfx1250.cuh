// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// gfx1250 a16w16 selector and strict kid dispatch.
#pragma once

#include "../opus_gemm_arch.cuh"
#include "../opus_gemm_common.cuh"
#include "../opus_gemm_utils.cuh"
#include "opus_gemm_heuristic_dispatch_gfx1250.cuh"
#include "opus_gemm_lookup.h"
#include "opus_gemm_a16w16_tune_lookup.h"
#include "opus_gemm_manifest.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
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

namespace opus_gfx1250_detail
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
        GENERATE_A16W16_WORKSPACE_TUNE_LOOKUP_GFX1250_SIZE>
        kWorkspace = {{GENERATE_A16W16_WORKSPACE_TUNE_LOOKUP_GFX1250}};
    return find_kid(kWorkspace, kid);
}

inline void check_shape_4g(int M, int N, int K, size_t c_element_size)
{
    constexpr uint64_t U32_MAX_BYTES = (1ULL << 32) - 1;
    const uint64_t a_bytes = static_cast<uint64_t>(M) * K * sizeof(bf16_t);
    const uint64_t b_bytes = static_cast<uint64_t>(N) * K * sizeof(bf16_t);
    const uint64_t c_bytes =
        static_cast<uint64_t>(M) * N * static_cast<uint64_t>(c_element_size);
    AITER_CHECK(a_bytes <= U32_MAX_BYTES && b_bytes <= U32_MAX_BYTES &&
                    c_bytes <= U32_MAX_BYTES,
                "opus gfx1250 a16w16 heuristic refuses >4 GiB shape (M=",
                M,
                " N=",
                N,
                " K=",
                K,
                "): launcher gmem descriptors are 32-bit");
}
} // namespace opus_gfx1250_detail

// gfx1250 currently has no non-workspace a16w16 kids.  These strict
// specializations still query the generated empty tables so a wrong-ABI call
// fails at the dispatch boundary instead of falling through to another arch.
template <typename CDataType>
inline OpusA16W16Kernel opus_a16w16_tune_dispatch_gfx1250(int id);

template <>
inline OpusA16W16Kernel opus_a16w16_tune_dispatch_gfx1250<bf16_t>(int id)
{
    using namespace opus_gfx1250_detail;
    static constexpr std::array<
        OpusA16W16TuneEntry, GENERATE_A16W16_TUNE_LOOKUP_GFX1250_BF16_SIZE>
        kTune = {{GENERATE_A16W16_TUNE_LOOKUP_GFX1250_BF16(bf16_t)}};
    const auto* entry = find_kid(kTune, id);
    AITER_CHECK(entry != nullptr,
                "Kernel id ",
                id,
                " not found in gfx1250 a16w16 bf16 non-workspace table");
    return entry->func;
}

template <>
inline OpusA16W16Kernel opus_a16w16_tune_dispatch_gfx1250<fp32_t>(int id)
{
    using namespace opus_gfx1250_detail;
    static constexpr std::array<
        OpusA16W16TuneEntry, GENERATE_A16W16_TUNE_LOOKUP_GFX1250_FP32_SIZE>
        kTune = {{GENERATE_A16W16_TUNE_LOOKUP_GFX1250_FP32(fp32_t)}};
    const auto* entry = find_kid(kTune, id);
    AITER_CHECK(entry != nullptr,
                "Kernel id ",
                id,
                " not found in gfx1250 a16w16 fp32 non-workspace table");
    return entry->func;
}

inline bool opus_a16w16_has_workspace_kernel_gfx1250(int id)
{
    return opus_gfx1250_detail::workspace_entry(id) != nullptr;
}

inline OpusA16W16WorkspaceKernel
opus_a16w16_workspace_dispatch_gfx1250(int id)
{
    const auto* entry = opus_gfx1250_detail::workspace_entry(id);
    AITER_CHECK(entry != nullptr,
                "Kernel id ",
                id,
                " not found in gfx1250 a16w16 workspace table");
    return entry->func;
}

template <typename CDataType>
inline int opus_select_a16w16_kid_gfx1250(
    int M, int N, int K, int batch, bool has_bias = false);

template <>
inline int opus_select_a16w16_kid_gfx1250<bf16_t>(
    int M, int N, int K, int batch, bool has_bias)
{
    using namespace opus_gfx1250_detail;
    static constexpr std::array<
        OpusA16W16RuntimeEntry, GENERATE_OPUS_LOOKUP_TABLE_GFX1250_BF16_SIZE>
        kLookup = {{GENERATE_OPUS_LOOKUP_TABLE_GFX1250_BF16}};
    const int tuned_kid = find_shape_kid(kLookup, M, N, K);
    if(tuned_kid >= 0) return tuned_kid;
    (void)batch;
    check_shape_4g(M, N, K, sizeof(bf16_t));
    return opus_a16w16_heuristic_kid_gfx1250(M, N, K, has_bias);
}

template <>
inline int opus_select_a16w16_kid_gfx1250<fp32_t>(
    int M, int N, int K, int batch, bool has_bias)
{
    using namespace opus_gfx1250_detail;
    static constexpr std::array<
        OpusA16W16RuntimeEntry, GENERATE_OPUS_LOOKUP_TABLE_GFX1250_FP32_SIZE>
        kLookup = {{GENERATE_OPUS_LOOKUP_TABLE_GFX1250_FP32}};
    const int tuned_kid = find_shape_kid(kLookup, M, N, K);
    if(tuned_kid >= 0) return tuned_kid;
    (void)batch;
    check_shape_4g(M, N, K, sizeof(fp32_t));
    return opus_a16w16_heuristic_kid_gfx1250(M, N, K, has_bias);
}
