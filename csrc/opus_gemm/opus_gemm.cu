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
#ifdef OPUS_BUILD_HAS_GFX950
#include "gfx950/opus_gemm_heuristic_dispatch_gfx950.cuh"  // OpusA16W16NoscaleKernel
#endif
#ifdef OPUS_BUILD_HAS_GFX942
#include "gfx942/opus_gemm_heuristic_dispatch_gfx942.cuh"
#endif
#include "opus_gemm_manifest.h"                    // a8w8 launcher symbols
#include "opus_gemm_a4w4_mxfp4_lookup.h"           // a4w4_mxfp4_nt kid table + tuned (M,N,K) -> kid
#include "gfx950/opus_co_launch_mxfp4_gfx950.cuh"  // opus_mxfp4_pick_split (host-only helpers)
#include "opus_gemm_utils.cuh"                     // bf16_t / fp32_t
#include "aiter_stream.h"                          // aiter::getCurrentHIPStream

#include <mutex>
#include <optional>
#include <unordered_map>

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

// ── a4w4_mxfp4_nt (gfx950, pre-compiled .co) ─────────────────────────────────
// Kid table (kid, launcher, B_M, B_N) and the tuned (M, N, K) -> kid table come from gen_instances.py.
struct OpusA4w4Mxfp4Kid {
  int kid;
  OpusScaleKernel fn;
  int b_m, b_n;
  int mode;       // 0 persistent, 1 one tile per block, 2 split-K, 3 small-tile (S = 1), 4 small-tile split-K (by kid only)
  int boundary;   // 0: needs M % B_M == 0 && N % B_N == 0
  int noshuffle;  // 1: a4w4_mxfp4_nt_noshuffle (row-major B / scales), 0: shuffle_weight B + e8m0_shuffle scales
};
struct OpusA4w4Mxfp4Tuned {
  int mnk[3];
  int kid;
};

static const OpusA4w4Mxfp4Kid* opus_a4w4_mxfp4_kids(int& n)
{
#if OPUS_A4W4_MXFP4_HAS_KIDS
  static const OpusA4w4Mxfp4Kid table[] = {GENERATE_A4W4_MXFP4_KIDS};
  n = static_cast<int>(sizeof(table) / sizeof(table[0]));
  return table;
#else
  n = 0;
  return nullptr;
#endif
}

static OpusScaleKernel opus_a4w4_mxfp4_by_kid(int kid)
{
  int n = 0;
  const auto* t = opus_a4w4_mxfp4_kids(n);
  for (int i = 0; i < n; ++i)
    if (t[i].kid == kid) return t[i].fn;
  AITER_CHECK(false, "opus a4w4_mxfp4: kid ", kid, " is not built");
  return nullptr;
}

// Heuristic: estimated time per (tile, variant), calibrated on clean HIP 6 measurements (gfx950, CUDA graph, 2026-11-13):
//   t = 3.2 us + rounds * B_M * B_N * K_seg * R(tile) * (1 + 0.5 * util)        util = busy blocks / CUs (power cap)
//   R: 256x256 6.7e-8, 192x256 / 256x192 7.35e-8, 256x128 / 128x256 7.5e-8, 128x128 8.3e-8 us per MAC-element
//   (fits 128x4096xK one-round np runs, 1024x4096x4096 full-util np and 8192^3 persistent within ~10%).
// Split-K (in-kernel last-block reduce): + 3.2 us + (S - 1) * 2.4 us * area / 16384 (the last block reads the other
// S - 1 fp32 partial tiles from one CU, ~25 GB/s); measured 128x128 128x4096x4096 S = 2 / 4 / 8: 12.3 / 15.5 / 24.7 us.
static int opus_a4w4_mxfp4_kid_noshuffle(int kid)
{
  int n = 0;
  const auto* t = opus_a4w4_mxfp4_kids(n);
  for (int i = 0; i < n; ++i)
    if (t[i].kid == kid) return t[i].noshuffle;
  AITER_CHECK(false, "opus a4w4_mxfp4: kid ", kid, " is not built");
  return 0;
}

static int opus_a4w4_mxfp4_heuristic_kid(int M, int N, int K, int batch, int noshuffle)
{
  int n = 0;
  const auto* t = opus_a4w4_mxfp4_kids(n);
  AITER_CHECK(n > 0, "opus a4w4_mxfp4: no kernels built");
  const int cus = static_cast<int>(get_num_cu_func());
  int best = -1;
  double best_cost = 1e300;
  // M <= 64: small-tile pipeline -- the first (kid order) S = 1 small kid of this layout with the smallest B_M >= M.
  if (M <= 64) {
    for (int i = 0; i < n; ++i) {
      const auto& e = t[i];
      if (e.mode != 3 || e.noshuffle != noshuffle || e.b_m < M) continue;
      if (best < 0 || e.b_m < best_cost) { best = e.kid; best_cost = e.b_m; }
    }
    if (best >= 0) return best;
    best_cost = 1e300;
  }
  for (int i = 0; i < n; ++i) {
    const auto& e = t[i];
    if (e.noshuffle != noshuffle || e.mode >= 3) continue;
    if (!e.boundary && (M % e.b_m || N % e.b_n)) continue;
    const long tiles = static_cast<long>((M + e.b_m - 1) / e.b_m) * ((N + e.b_n - 1) / e.b_n) * batch;
    const double area = static_cast<double>(e.b_m) * e.b_n;
    const double R = area >= 65536 ? 6.7e-8 : area >= 49152 ? 7.35e-8 : area >= 32768 ? 7.5e-8 : 8.3e-8;
    double cost;
    if (e.mode == 0) {
      if (tiles <= cus) continue;   // single round: the one-tile-per-block variant does the same without cross-tile code
      const long rounds = (tiles + cus - 1) / cus;
      cost = 3.2 + rounds * area * K * R * 1.5;
      if (e.boundary) cost *= 1.003;   // prefer the no-boundary variant when it applies
    } else if (e.mode == 1) {
      if (tiles > cus) continue;
      cost = 3.2 + area * K * R * (1.0 + 0.5 * static_cast<double>(tiles) / cus);
    } else {
      const int S = opus_mxfp4_pick_split(static_cast<int>(tiles), K, cus);
      if (S <= 1) continue;
      cost = 3.2 + area * (K / S) * R * (1.0 + 0.5 * static_cast<double>(tiles * S) / cus)
           + 3.2 + (S - 1) * 2.4 * area / 16384.0;
    }
    if (cost < best_cost) { best_cost = cost; best = e.kid; }
  }
  AITER_CHECK(best >= 0, "opus a4w4_mxfp4: no applicable kernel");
  return best;
}

static OpusScaleKernel opus_dispatch_a4w4_mxfp4_gfx950(int M, int N, int K, int batch, int noshuffle)
{
  static const OpusA4w4Mxfp4Tuned tuned[] = {GENERATE_A4W4_MXFP4_TUNED {{0, 0, 0}, -1}};
  for (const auto& e : tuned)
    if (e.kid >= 0 && e.mnk[0] == M && e.mnk[1] == N && e.mnk[2] == K && opus_a4w4_mxfp4_kid_noshuffle(e.kid) == noshuffle)
      return opus_a4w4_mxfp4_by_kid(e.kid);
  return opus_a4w4_mxfp4_by_kid(opus_a4w4_mxfp4_heuristic_kid(M, N, K, batch, noshuffle));
}

static void opus_a4w4_mxfp4_check(const aiter_tensor_t& XQ, const aiter_tensor_t& WQ, const aiter_tensor_t& Y,
                                  const std::optional<aiter_tensor_t>& x_scale,
                                  const std::optional<aiter_tensor_t>& w_scale,
                                  int noshuffle = 0)
{
  const auto& arch_info = opus_get_arch_info();
  AITER_CHECK(arch_info.arch == OpusGfxArch::Gfx950, "opus a4w4_mxfp4 is gfx950-only; device ", arch_info.name);
  AITER_CHECK(WQ.dtype() == AITER_DTYPE_fp4x2, "opus a4w4_mxfp4: WQ must be fp4x2");
  AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16, "opus a4w4_mxfp4: Y must be bf16");
  AITER_CHECK(x_scale.has_value() && w_scale.has_value(), "opus a4w4_mxfp4 needs x_scale and w_scale (e8m0_shuffle)");
  AITER_CHECK(XQ.is_contiguous() && WQ.is_contiguous() && Y.is_contiguous() &&
              x_scale.value().is_contiguous() && w_scale.value().is_contiguous(),
              "opus a4w4_mxfp4: tensors must be contiguous");
  // Shapes: XQ [batch, M, K/2], WQ [batch, N, K/2] (fp4x2), Y [batch, M, N]; N % 16 (shuffle_weight), K % 256 (K tile).
  const int64_t batch = XQ.size(0), M = XQ.size(1), K = XQ.size(2) * 2, N = WQ.size(1);
  AITER_CHECK(WQ.size(0) == batch && WQ.size(2) * 2 == K, "opus a4w4_mxfp4: WQ must be [batch, N, K/2] matching XQ");
  AITER_CHECK(Y.size(0) == batch && Y.size(1) == M && Y.size(2) == N, "opus a4w4_mxfp4: Y must be [batch, M, N]");
  AITER_CHECK(K % 256 == 0 && K > 0, "opus a4w4_mxfp4: K must be a multiple of 256 (got K=", K, ")");
  if (noshuffle) {
    // Row-major B: N % 8 (16 B C stores); row-major scales [rows, K/32] per batch, 2D [batch * rows, K/32] or 3D.
    AITER_CHECK(N % 8 == 0, "opus a4w4_mxfp4 (bpreshuffle=False): N must be a multiple of 8 (got N=", N, ")");
    auto check_rm = [&](const aiter_tensor_t& s, int64_t rows, const char* nm) {
      AITER_CHECK(s.element_size() == 1, "opus a4w4_mxfp4: ", nm, " must be e8m0 (1 byte)");
      AITER_CHECK(s.dim() == 2 || (s.dim() == 3 && s.size(0) == batch), "opus a4w4_mxfp4: ", nm, " must be 2D or [batch, rows, K/32]");
      AITER_CHECK(s.dim() == 3 || s.size(0) % batch == 0, "opus a4w4_mxfp4: ", nm, " rows not divisible by batch");
      const int64_t r = s.dim() == 3 ? s.size(1) : s.size(0) / batch;
      AITER_CHECK(s.size(-1) == K / 32, "opus a4w4_mxfp4 (bpreshuffle=False): ", nm, " must be row-major with K/32 = ", K / 32,
                  " columns (got ", s.size(-1), ")");
      AITER_CHECK(r >= rows, "opus a4w4_mxfp4 (bpreshuffle=False): ", nm, " needs >= ", rows, " rows per batch (got ", r, ")");
    };
    check_rm(x_scale.value(), M, "x_scale");
    check_rm(w_scale.value(), N, "w_scale");
    return;
  }
  AITER_CHECK(N % 16 == 0, "opus a4w4_mxfp4: N must be a multiple of 16 (got N=", N, ")");
  // Scales: e8m0_shuffle layout, per batch [rows padded to >= 32, K/32 padded to 8]; 2D [batch * rows, cols] or 3D.
  const int64_t cols = (K / 32 + 7) / 8 * 8;
  auto check_sf = [&](const aiter_tensor_t& s, int64_t rows, const char* nm) {
    AITER_CHECK(s.element_size() == 1, "opus a4w4_mxfp4: ", nm, " must be e8m0 (1 byte)");
    AITER_CHECK(s.dim() == 2 || (s.dim() == 3 && s.size(0) == batch), "opus a4w4_mxfp4: ", nm, " must be 2D or [batch, rows, cols]");
    AITER_CHECK(s.dim() == 3 || s.size(0) % batch == 0, "opus a4w4_mxfp4: ", nm, " rows not divisible by batch");
    const int64_t r = s.dim() == 3 ? s.size(1) : s.size(0) / batch;
    AITER_CHECK(s.size(-1) == cols, "opus a4w4_mxfp4: ", nm, " must have K/32 padded to 8 = ", cols, " columns (got ", s.size(-1), ")");
    AITER_CHECK(r >= (rows + 31) / 32 * 32 && r % 32 == 0, "opus a4w4_mxfp4: ", nm, " needs >= ", (rows + 31) / 32 * 32,
                " rows (multiple of 32, e8m0_shuffle) per batch (got ", r, ")");
  };
  check_sf(x_scale.value(), M, "x_scale");
  check_sf(w_scale.value(), N, "w_scale");
}

// a16w16 arch routers: switch on opus_get_gfx_arch() to per-arch dispatch.
template <typename CDataType>
OpusA16W16NoscaleKernel opus_dispatch_a16w16(int M, int N, int K, int batch, bool has_bias = false)
{
  switch (opus_get_gfx_arch())
  {
#ifdef OPUS_BUILD_HAS_GFX950
    case OpusGfxArch::Gfx950:
      return opus_dispatch_a16w16_gfx950<CDataType>(M, N, K, batch, has_bias);
#endif
#ifdef OPUS_BUILD_HAS_GFX942
    case OpusGfxArch::Gfx942:
      return opus_dispatch_a16w16_gfx942<CDataType>(M, N, K, batch, has_bias);
#endif
#ifdef OPUS_BUILD_HAS_GFX1250
    case OpusGfxArch::Gfx1250:
      return opus_dispatch_a16w16_gfx1250<CDataType>(M, N, K, batch, has_bias);
#endif
    default:
    {
      const auto &info = opus_get_arch_info();
      AITER_CHECK(false,
                  "opus_gemm: a16w16 dispatch is only implemented for gfx950 today; "
                  "current device ", info.dev,
                  " has gcnArchName='", info.name,
                  "'. Other archs (gfx940 / gfx942 / gfx1100 / ...) will be added "
                  "as more pipelines land.");
    }
  }
}

template <typename CDataType>
OpusA16W16NoscaleKernel
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
                  "opus_gemm_a16w16_tune: dispatch is only implemented for gfx950 today; "
                  "current device ", info.dev,
                  " has gcnArchName='", info.name,
                  "'. Other archs will be added as more pipelines land.");
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
    // Tuned-lookup-then-heuristic dispatch. splitK=0 = "launcher decides".
    int batch = XQ.size(0);
    const bool has_bias = bias.has_value();
    if (Y.dtype() == AITER_DTYPE_bf16)
    {
      opus_dispatch_a16w16<bf16_t>(M, N, K, batch, has_bias)(XQ, WQ, Y, bias, 0);
    }
    else if (Y.dtype() == AITER_DTYPE_fp32)
    {
      opus_dispatch_a16w16<fp32_t>(M, N, K, batch, has_bias)(XQ, WQ, Y, bias, 0);
    }
    else
    {
      AITER_CHECK(false, "opus_gemm a16w16: unsupported output dtype, expected bf16 or fp32");
    }
  }
  else if (XQ.dtype() == AITER_DTYPE_fp4x2)
  {
#ifdef OPUS_BUILD_HAS_GFX950
    AITER_CHECK(!bias.has_value(), "opus_gemm a4w4_mxfp4: bias is not supported");
    opus_a4w4_mxfp4_check(XQ, WQ, Y, x_scale, w_scale);   // opus_gemm(): bpreshuffle layout (see opus_gemm_a4w4)
    opus_dispatch_a4w4_mxfp4_gfx950(M, N, K * 2, XQ.size(0), 0)(XQ, WQ, Y, x_scale, w_scale);
#else
    AITER_CHECK(false, "opus_gemm a4w4_mxfp4 needs a gfx950 build of module_deepgemm_opus");
#endif
  }
  else
  {
    AITER_CHECK(false, "opus_gemm: unsupported input dtype, expected fp8, bf16 or fp4x2");
  }
}

// opus_gemm_a4w4(): MXFP4 GEMM with an explicit input layout, as aiter gemm_a4w4: bpreshuffle = true -> B shuffle_weight(16,16)
// + e8m0_shuffle scales (a4w4_mxfp4_nt), false -> plain row-major B and [rows, K/32] scales (a4w4_mxfp4_nt_noshuffle).
void opus_gemm_a4w4(
  aiter_tensor_t &XQ,
  aiter_tensor_t &WQ,
  aiter_tensor_t &Y,
  std::optional<aiter_tensor_t> x_scale,
  std::optional<aiter_tensor_t> w_scale,
  bool bpreshuffle)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(XQ.dim() == 3 && WQ.dim() == 3 && Y.dim() == 3, "opus_gemm_a4w4: XQ / WQ / Y must be 3D");
  AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp4x2, "opus_gemm_a4w4: XQ must be fp4x2");
#ifdef OPUS_BUILD_HAS_GFX950
  const int noshuffle = bpreshuffle ? 0 : 1;
  opus_a4w4_mxfp4_check(XQ, WQ, Y, x_scale, w_scale, noshuffle);
  opus_dispatch_a4w4_mxfp4_gfx950(XQ.size(1), WQ.size(1), XQ.size(2) * 2, XQ.size(0), noshuffle)(XQ, WQ, Y, x_scale, w_scale);
#else
  (void)bpreshuffle;
  AITER_CHECK(false, "opus_gemm_a4w4 needs a gfx950 build of module_deepgemm_opus");
#endif
}

// opus_gemm_a4w4_tune(): run a specific a4w4_mxfp4_nt(_noshuffle) kid (kernelId = kid: 300xx shuffle, 301xx noshuffle);
// the input layout follows the kid.
void opus_gemm_a4w4_tune(
  aiter_tensor_t &XQ,
  aiter_tensor_t &WQ,
  aiter_tensor_t &Y,
  std::optional<aiter_tensor_t> x_scale,
  std::optional<aiter_tensor_t> w_scale,
  int kernelId)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(XQ.dim() == 3 && WQ.dim() == 3 && Y.dim() == 3, "opus_gemm_a4w4_tune: XQ / WQ / Y must be 3D");
  AITER_CHECK(XQ.dtype() == AITER_DTYPE_fp4x2, "opus_gemm_a4w4_tune: XQ must be fp4x2");
#ifdef OPUS_BUILD_HAS_GFX950
  opus_a4w4_mxfp4_check(XQ, WQ, Y, x_scale, w_scale, opus_a4w4_mxfp4_kid_noshuffle(kernelId));
  opus_a4w4_mxfp4_by_kid(kernelId)(XQ, WQ, Y, x_scale, w_scale);
#else
  AITER_CHECK(false, "opus_gemm_a4w4_tune needs a gfx950 build of module_deepgemm_opus");
#endif
}

// opus_gemm_a16w16_tune() — id-based tune entry.

// splitk kids: gfx950 [200,300) + nooob [1200,1300); gfx942 [10200, 10300).
static constexpr int OPUS_SPLITK_KID_MIN = 200;
static constexpr int OPUS_SPLITK_KID_MAX = 300;
static constexpr int OPUS_GFX942_KID_OFFSET = 10000;
static constexpr int OPUS_GFX942_SPLITK_KID_MAX = 300;
// gfx1250 cluster/TDM split-K (fp32 workspace + reduce) kids: [20000, 21000).
static constexpr int OPUS_GFX1250_SPLITK_KID_MIN = 20000;
static constexpr int OPUS_GFX1250_SPLITK_KID_MAX = 21000;
// SB a16w16 kids: gfx950 [4,10) + mirrors at +1000/.../+7000.
static constexpr int OPUS_A16W16_SB_KID_MIN = 4;
static constexpr int OPUS_A16W16_SB_KID_MAX = 10;
// Persistent a16w16 kids: compact [300, 316) = 4 tiles × 4 cpol groups.
static constexpr int OPUS_PERSISTENT_KID_MIN = 300;
static constexpr int OPUS_PERSISTENT_KID_MAX = 316;
// Mono-tile a16w16 kids: [1400, 1500). Mono-tile is intrinsically non-OOB
// (no tail handling in the kernel body), so kids land in the >=1000 band
// directly — there is no base/nooob mirror split for this family. See
// opus_gemm_common.py :: a16w16_mono_tile_kernels_list.
static constexpr int OPUS_MONO_TILE_KID_MIN = 1400;
static constexpr int OPUS_MONO_TILE_KID_MAX = 1500;
// non-OOB kid offset
static constexpr int OPUS_NOOOB_KID_OFFSET = 1000;

static inline bool opus_kid_is_splitk(int kid)
{
  return (kid >= OPUS_SPLITK_KID_MIN && kid < OPUS_SPLITK_KID_MAX) ||
         (kid >= OPUS_SPLITK_KID_MIN + OPUS_NOOOB_KID_OFFSET &&
          kid < OPUS_SPLITK_KID_MAX + OPUS_NOOOB_KID_OFFSET) ||
         (kid >= OPUS_SPLITK_KID_MIN + OPUS_GFX942_KID_OFFSET &&
          kid < OPUS_GFX942_SPLITK_KID_MAX + OPUS_GFX942_KID_OFFSET) ||
         (kid >= OPUS_GFX1250_SPLITK_KID_MIN &&
          kid < OPUS_GFX1250_SPLITK_KID_MAX);
}

static inline bool opus_kid_is_a16w16_sb(int kid)
{
  // SB a16w16 kid bases: 0/1000/2000/.../7000 + [4,10) (cpol mirrors).
  for (int base : {0, 1000, 2000, 3000, 4000, 5000, 6000, 7000})
  {
    if (kid >= base + OPUS_A16W16_SB_KID_MIN && kid < base + OPUS_A16W16_SB_KID_MAX)
      return true;
  }
  return false;
}

static inline bool opus_kid_is_persistent(int kid)
{
  return (kid >= OPUS_PERSISTENT_KID_MIN && kid < OPUS_PERSISTENT_KID_MAX) ||
         (kid >= OPUS_PERSISTENT_KID_MIN + OPUS_NOOOB_KID_OFFSET &&
          kid < OPUS_PERSISTENT_KID_MAX + OPUS_NOOOB_KID_OFFSET);
}

static inline bool opus_kid_is_mono_tile(int kid)
{
  // Mono-tile lives entirely in the non-OOB band [1400, 1500); no mirror.
  return kid >= OPUS_MONO_TILE_KID_MIN && kid < OPUS_MONO_TILE_KID_MAX;
}

static inline bool opus_kid_is_gfx942_splitk(int kid)
{
  return kid >= OPUS_SPLITK_KID_MIN + OPUS_GFX942_KID_OFFSET &&
         kid < OPUS_GFX942_SPLITK_KID_MAX + OPUS_GFX942_KID_OFFSET;
}

static inline bool opus_kid_is_gfx1250_splitk(int kid)
{
  return kid >= OPUS_GFX1250_SPLITK_KID_MIN && kid < OPUS_GFX1250_SPLITK_KID_MAX;
}

static inline bool opus_kid_supports_bias(int kid)
{
  // persistent and mono-tile do not support bias (kargs lacks
  // ptr_bias/stride_bias_batch; launchers reject non-empty bias up front).
  // gfx942 splitk/SB silently ignored bias; exclude explicitly to surface
  // misuse as a clear error.
  // gfx1250 cluster_tdm_splitk_ws DOES support bias (the reduce kernel folds
  // it once, like gfx950 flatmm_splitk).
  return (opus_kid_is_a16w16_sb(kid) || opus_kid_is_splitk(kid))
         && !opus_kid_is_gfx942_splitk(kid);
}

void opus_gemm_a16w16_tune(
    aiter_tensor_t &XQ,
    aiter_tensor_t &WQ,
    aiter_tensor_t &Y,
    std::optional<aiter_tensor_t> bias,
    int kernelId,
    int splitK)
{
  aiter_detail::g_aiter_can_throw = true;
  AITER_CHECK(XQ.dim() == 3, "XQ must be 3D [batch, M, K]");
  AITER_CHECK(WQ.dim() == 3, "WQ must be 3D [batch, N, K]");
  AITER_CHECK(Y.dim() == 3, "Y must be 3D [batch, M, N]");
  AITER_CHECK(XQ.dtype() == WQ.dtype(),
              "XQ and WQ should have the same dtype!");
  // Early-gate non-bias-capable kids for a clean error before launcher entry.
  AITER_CHECK(!bias.has_value() || opus_kid_supports_bias(kernelId),
              "opus_gemm_a16w16_tune: bias is currently only supported on "
              "a16w16 split-barrier kids [", OPUS_A16W16_SB_KID_MIN, ", ",
              OPUS_A16W16_SB_KID_MAX, ") or a16w16_flatmm_splitk kids [",
              OPUS_SPLITK_KID_MIN, ", ", OPUS_SPLITK_KID_MAX,
              "); got kid=", kernelId);

  if (XQ.dtype() == AITER_DTYPE_bf16)
  {
    // All splitk kids (gfx950/gfx942/gfx1250) force <fp32_t>: the main kernel
    // writes an fp32 workspace and a reduce kernel casts it to Y.dtype() at
    // runtime (gfx1250 cluster_tdm_splitk_ws now follows this same pattern).
    if (opus_kid_is_splitk(kernelId))
    {
      AITER_CHECK(Y.dtype() == AITER_DTYPE_bf16
                  || Y.dtype() == AITER_DTYPE_fp32,
                  "opus_gemm_a16w16_tune splitk kid requires bf16 or fp32 Y "
                  "(reduce kernel writes the correct dtype)");
      opus_a16w16_tune_dispatch<fp32_t>(kernelId)(XQ, WQ, Y, bias, splitK);
    }
    else if (Y.dtype() == AITER_DTYPE_bf16)
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

// ──────────────────────────────────────────────────────────────────────────────
// Splitk fp32 workspace: per-stream owner.
//
// Each splitk launcher (generated by gen_instances.py) needs a stable
// `opus_splitk_ws_handle*` to feed into both the main kernel and the reduce
// kernel; captured HIP graphs bake in that pointer. Previously this was a
// `static thread_local` slot — one handle per CPU thread — but under
// vLLM/sglang-style TBO two CPU threads drive two streams concurrently, and
// each captured graph needs its own buffer pointer baked in. The TLS form
// also tripped the in-capture grow guard on the second thread.
//
// Now we own the handle by stream: a process-global mutex-protected map
// keyed by hipStream_t. Eager: lazy-create on first lookup. Capture: caller
// must pre-register the handle via opus_gemm_workspace_init(), otherwise the
// lookup throws (cleaner than the prior SIGABRT). The framework calls
// opus_gemm_workspace_init() once per TBO stream eagerly before capture.
namespace {
struct SplitkWsRegistry {
  std::mutex mu;
  struct Owner {
    opus_splitk_ws_handle* host;
    opus_splitk_ws_handle* device;
  };
  std::unordered_map<hipStream_t, Owner*> map;
};
SplitkWsRegistry& splitk_ws_registry()
{
  static SplitkWsRegistry r;
  return r;
}
} // anonymous

opus_splitk_ws_handle* opus_splitk_ws_get(hipStream_t s, bool allow_create)
{
  auto& R = splitk_ws_registry();
  std::lock_guard<std::mutex> g(R.mu);
  auto it = R.map.find(s);
  if (it != R.map.end()) return it->second->host;
  AITER_CHECK(allow_create,
              "splitk workspace not initialized for the current CUDA stream. "
              "Call aiter.opus_gemm_workspace_init() inside "
              "`with torch.cuda.stream(s):` (and warm with the largest "
              "expected gemm) before HIP graph capture.");
  auto* owner = new SplitkWsRegistry::Owner{};
  opus_splitk_ws_handle* h = nullptr;
  HIP_CALL(hipHostMalloc(reinterpret_cast<void**>(&h),
                         sizeof(opus_splitk_ws_handle),
                         hipHostMallocCoherent));
  h->ptr   = nullptr;
  h->bytes = 0;
  owner->host   = h;
  owner->device = nullptr;
  R.map[s]      = owner;
  return h;
}

const opus_splitk_ws_handle* opus_splitk_ws_device_handle(hipStream_t s, bool allow_create)
{
  (void)opus_splitk_ws_get(s, allow_create);
  auto& R = splitk_ws_registry();
  std::lock_guard<std::mutex> g(R.mu);
  auto it = R.map.find(s);
  AITER_CHECK(it != R.map.end(), "splitk workspace not initialized for the current CUDA stream.");
  if (it->second->device == nullptr)
  {
    AITER_CHECK(allow_create,
                "splitk workspace device handle not initialized for the current CUDA stream. "
                "Warm the opus gfx942 splitK launcher eagerly before HIP graph capture.");
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&it->second->device),
                       sizeof(opus_splitk_ws_handle)));
    HIP_CALL(hipMemcpy(it->second->device,
                       it->second->host,
                       sizeof(opus_splitk_ws_handle),
                       hipMemcpyHostToDevice));
  }
  return it->second->device;
}

void opus_splitk_ws_sync_to_device(hipStream_t s)
{
  auto& R = splitk_ws_registry();
  std::lock_guard<std::mutex> g(R.mu);
  auto it = R.map.find(s);
  AITER_CHECK(it != R.map.end(), "splitk workspace not initialized for the current CUDA stream.");
  if (it->second->device == nullptr)
  {
    HIP_CALL(hipMalloc(reinterpret_cast<void**>(&it->second->device),
                       sizeof(opus_splitk_ws_handle)));
  }
  HIP_CALL(hipMemcpy(it->second->device,
                     it->second->host,
                     sizeof(opus_splitk_ws_handle),
                     hipMemcpyHostToDevice));
}

void opus_gemm_workspace_init()
{
  hipStream_t s = aiter::getCurrentHIPStream();
  hipStreamCaptureStatus cap = hipStreamCaptureStatusNone;
  HIP_CALL(hipStreamIsCapturing(s, &cap));
  AITER_CHECK(cap == hipStreamCaptureStatusNone,
              "opus_gemm_workspace_init must be called in eager mode "
              "(not inside HIP graph capture).");
  (void)opus_splitk_ws_get(s, /*allow_create=*/true);
}

#endif // !__HIP_DEVICE_COMPILE__
