// SPDX-License-Identifier: MIT
// Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

// Host-side dispatcher (lookup table + heuristic).
#ifndef __HIP_DEVICE_COMPILE__

#include "opus_gemm_arch.cuh"                      // OpusGfxArch + opus_get_arch_info / opus_get_gfx_arch
#include "opus_build_archs.h"                      // OPUS_BUILD_HAS_GFX942 / OPUS_BUILD_HAS_GFX950
#ifdef OPUS_BUILD_HAS_GFX950
#include "gfx950/opus_gemm_arch_gfx950.cuh"        // opus_dispatch_a16w16_gfx950<T> / opus_a16w16_tune_dispatch_gfx950<T>
#endif
#ifdef OPUS_BUILD_HAS_GFX942
#include "gfx942/opus_gemm_arch_gfx942.cuh"        // opus_dispatch_a16w16_gfx942<T> / opus_a16w16_tune_dispatch_gfx942<T>
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
#include "gfx1250/opus_gemm_arch_gfx1250.cuh"      // opus_a16w16_tune_dispatch_gfx1250<T> (tune-id entry only)
#endif
#include "opus_gemm_common.cuh"
#include "opus_gemm_manifest.h"                    // a8w8 launcher symbols
#include "opus_gemm_utils.cuh"                     // bf16_t / fp32_t
#include "aiter_stream.h"                          // aiter::getCurrentHIPStream

#include <optional>

// a8w8 / a8w8_scale: single hardcoded launcher per dtype (no tuned table).
// Plain fn ptrs; std::function's type-erasure is pure waste here.
using OpusScaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &,
    std::optional<aiter_tensor_t>, std::optional<aiter_tensor_t>);

using OpusNoscaleKernel = void (*)(
    aiter_tensor_t &, aiter_tensor_t &,
    aiter_tensor_t &);

template <typename CDataType>
OpusScaleKernel opus_dispatch_scale(int M, int N, int K)
{
#ifdef OPUS_BUILD_HAS_GFX950
  return opus_gemm_512x256x256x128_4x2_16x16x128_1x128x128<CDataType>;
#else
  (void)M;
  (void)N;
  (void)K;
  return nullptr;
#endif
}

template <typename CDataType>
OpusNoscaleKernel opus_dispatch_a8w8(int M, int N, int K)
{
#ifdef OPUS_BUILD_HAS_GFX950
  return opus_gemm_512x256x256x128_2x4_16x16x128_0x0x0<CDataType>;
#else
  (void)M;
  (void)N;
  (void)K;
  return nullptr;
#endif
}

template <typename CDataType>
static OpusA16W16Kernel
opus_a16w16_tune_dispatch(int id)
{
  switch (opus_get_gfx_arch())
  {
#ifdef OPUS_BUILD_HAS_GFX950
    case OpusGfxArch::Gfx950:
      return opus_a16w16_tune_dispatch_gfx950<CDataType>(id);
#endif
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_a16w16_tune_dispatch_gfx942<CDataType>(id);
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
    case OpusGfxArch::Gfx1250:
      return opus_a16w16_tune_dispatch_gfx1250<CDataType>(id);
#endif
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm_a16w16_tune: no non-workspace dispatch table for "
                  "current device ", info.dev,
                  " with gcnArchName='", info.name, "'");
      return nullptr;
    }
  }
}

// The generated per-arch workspace tables are the sole source of truth for
// the five-argument vs. six-argument launcher ABI.  Keep this routing next to
// the strict non-workspace router so an id from another architecture cannot
// accidentally match a copied numeric range.
static bool opus_a16w16_has_workspace_kernel(int id)
{
  switch (opus_get_gfx_arch())
  {
#ifdef OPUS_BUILD_HAS_GFX950
    case OpusGfxArch::Gfx950:
      return opus_a16w16_has_workspace_kernel_gfx950(id);
#endif
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_a16w16_has_workspace_kernel_gfx942(id);
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
    case OpusGfxArch::Gfx1250:
      return opus_a16w16_has_workspace_kernel_gfx1250(id);
#endif
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm_a16w16_tune: no workspace dispatch table for device ",
                  info.dev, " with gcnArchName='", info.name, "'");
      return false;
    }
  }
}

static OpusA16W16WorkspaceKernel opus_a16w16_workspace_dispatch(int id)
{
  switch (opus_get_gfx_arch())
  {
#ifdef OPUS_BUILD_HAS_GFX950
    case OpusGfxArch::Gfx950:
      return opus_a16w16_workspace_dispatch_gfx950(id);
#endif
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_a16w16_workspace_dispatch_gfx942(id);
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
    case OpusGfxArch::Gfx1250:
      return opus_a16w16_workspace_dispatch_gfx1250(id);
#endif
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm_a16w16_tune: no workspace dispatch table for device ",
                  info.dev, " with gcnArchName='", info.name, "'");
      return nullptr;
    }
  }
}

// ── opus_gemm() — top-level a16w16 / a8w8 entry ─────────────────────────────

void opus_gemm(
  aiter_tensor_t &XQ,
  aiter_tensor_t &WQ,
  aiter_tensor_t &Y,
  std::optional<aiter_tensor_t> group_layout,
  std::optional<aiter_tensor_t> x_scale,
  std::optional<aiter_tensor_t> w_scale,
  std::optional<aiter_tensor_t> bias)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(XQ.dim() == 3, "XQ must be 3D [batch, M, K]");
  AITER_CHECK(WQ.dim() == 3, "WQ must be 3D [batch, N, K]");
  AITER_CHECK(Y.dim() == 3, "Y must be 3D [batch, M, N]");

  int M = XQ.size(1);
  int N = WQ.size(1);
  int K = XQ.size(2);

  bool has_scale = x_scale.has_value() && w_scale.has_value();

  if (XQ.dtype() == AITER_DTYPE_fp8)
  {
    // a8w8 / a8w8_scale launchers are gfx950-only today and don't yet flow through the arch-routed
    // dispatcher (they pick a single har...
    const auto &arch_info = opus_get_arch_info();
#ifdef OPUS_BUILD_HAS_GFX950
    AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950,
                "opus_gemm: a8w8 path is only implemented for gfx950 today; "
                "current device ", arch_info.dev,
                " has gcnArchName='", arch_info.name,
                "'. Other archs will be added as more pipelines land.");
    // a8w8 / a8w8_scale launchers do not consume bias yet; reject up front
    // rather than silently dropping it.
    AITER_CHECK(!bias.has_value(),
                "opus_gemm: bias is not supported on a8w8 / a8w8_scale paths");
    if (has_scale)
    {
      AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32,
                  "opus_gemm a8w8_scale only supports fp32 output");
      opus_dispatch_scale<fp32_t>(M, N, K)(XQ, WQ, Y, x_scale, w_scale);
    }
    else
    {
      AITER_CHECK(Y.dtype() == AITER_DTYPE_fp32,
                  "opus_gemm a8w8 no-scale only supports fp32 output");
      opus_dispatch_a8w8<fp32_t>(M, N, K)(XQ, WQ, Y);
    }
#else
    AITER_CHECK(false,
                "opus_gemm: a8w8 path requires module_deepgemm_opus to be "
                "built with OPUS_BUILD_HAS_GFX950; current device ",
                arch_info.dev, " has gcnArchName='", arch_info.name, "'");
#endif
  }
  else if (XQ.dtype() == AITER_DTYPE_bf16)
  {
    AITER_CHECK(false,
                "opus_gemm: generic bf16 a16w16 dispatch is disabled; use "
                "aiter.ops.opus.gemm_a16w16_opus or "
                "opus_gemm_a16w16_tune so Python resolves the actual kernel "
                "and supplies its typed Torch workspace");
  }
  else
  {
    AITER_CHECK(false, "opus_gemm: unsupported input dtype, expected fp8 or bf16");
  }
}

// opus_gemm_a16w16_tune() — id-based tune entry.

void opus_gemm_a16w16_tune(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    std::optional<aiter_tensor_t> workspace,
    int kernelId,
    int splitK)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(XQ.dim() == 3, "XQ must be 3D [batch, M, K]");
  AITER_CHECK(WQ.dim() == 3, "WQ must be 3D [batch, N, K]");
  AITER_CHECK(Y.dim() == 3, "Y must be 3D [batch, M, N]");
  AITER_CHECK(XQ.dtype() == WQ.dtype(),
              "XQ and WQ should have the same dtype!");

  if (XQ.dtype() == AITER_DTYPE_bf16)
  {
    const bool uses_workspace = opus_a16w16_has_workspace_kernel(kernelId);
    if (uses_workspace)
    {
      AITER_CHECK(workspace.has_value(),
                  "opus_gemm_a16w16_tune: workspace kernel id ", kernelId,
                  " requires a workspace tensor");
      AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16
                  || Y.dtype() == AITER_DTYPE_fp32,
                  "opus_gemm_a16w16_tune workspace kid requires bf16 or fp32 Y "
                  "(reduce kernel writes the correct dtype)");
      opus_a16w16_workspace_dispatch(kernelId)(
          XQ, WQ, Y, workspace.value(), bias, splitK);
    }
    else
    {
      AITER_CHECK(!workspace.has_value(),
                  "opus_gemm_a16w16_tune: non-workspace kernel id ", kernelId,
                  " requires workspace=None");
      if (Y.dtype() == AITER_DTYPE_bf16)
      {
        opus_a16w16_tune_dispatch<bf16_t>(kernelId)(XQ, WQ, Y, bias, splitK);
      }
      else if (Y.dtype() == AITER_DTYPE_fp32)
      {
        opus_a16w16_tune_dispatch<fp32_t>(kernelId)(XQ, WQ, Y, bias, splitK);
      }
      else
      {
        AITER_CHECK(false,
                    "opus_gemm_a16w16_tune: unsupported output dtype, expected bf16 or fp32");
      }
    }
  }
  else
  {
    AITER_CHECK(false,
                "opus_gemm_a16w16_tune: unsupported input dtype ",
                AiterDtype_to_str(XQ.dtype()),
                ", expected bf16");
  }
}

void opus_gemm_a8w8_blockscale_bpreshuffle_tune(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    std::optional<aiter_tensor_t> x_scale,
    std::optional<aiter_tensor_t> w_scale,
    aiter_tensor_t &Y,
    int kernelId)
{
  aiter_detail::g_aiter_can_throw = true;
  const auto &arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx942,
              "opus_gemm_a8w8_blockscale_bpreshuffle_tune is only implemented "
              "for gfx942 today; current device ", arch_info.dev,
              " has gcnArchName='", arch_info.name, "'");
  AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp8 && WQ.dtype() == AITER_DTYPE_fp8,
              "opus_gemm_a8w8_blockscale_bpreshuffle_tune expects fp8 XQ/WQ");
  AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16,
              "opus_gemm_a8w8_blockscale_bpreshuffle_tune expects bf16 Y");
  AITER_CHECK(x_scale.has_value() && w_scale.has_value(),
              "opus_gemm_a8w8_blockscale_bpreshuffle_tune requires x_scale and w_scale");

#ifdef OPUS_BUILD_HAS_GFX942
  opus_a8w8_tune_dispatch_gfx942(kernelId)(XQ, WQ, Y, x_scale, w_scale);
#else
  AITER_CHECK(false,
              "module_deepgemm_opus was not built with OPUS_BUILD_HAS_GFX942");
#endif
}

#endif // !__HIP_DEVICE_COMPILE__
