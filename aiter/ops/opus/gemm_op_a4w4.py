# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus gfx950 MXFP4 x MXFP4 NT GEMM (1x32 e8m0 scales) with plain row-major inputs (kernel a4w4_mxfp4_nt_noshuffle),
the layout of Triton ``gemm_afp4wfp4``:
  * ``XQ``: [M, K/2] (or [batch, M, K/2]) fp4x2 or uint8, row-major.
  * ``WQ``: [N, K/2] (or [batch, N, K/2]) fp4x2 or uint8, row-major.
  * ``x_scale`` / ``w_scale``: [M, K/32] / [N, K/32] e8m0 or uint8, row-major (2D [batch * rows, K/32] or 3D per batch).
  Any M; N % 8 == 0; K % 256 == 0.

Output bf16 [M, N] (or [batch, M, N]).  Device code: the 2x2 kids are the pre-compiled gen_co/gfx950/*.co (pin
toolchain), the small-tile kids are JIT-built.
"""

import torch
from torch import Tensor

import aiter

from ...jit.core import compile_ops

# kid = 30100 + variant offset + tile index; tiles 256x256, 256x128, 128x256, 128x128, 192x256, 256x192.
# variants: +0 persistent, +10 one tile per block, 30130 = persistent 256x256 without M/N boundary handling (M, N multiples
# of 256).  Small-tile (1x4 waves, A / B / scales through an LDS ring) 30500 + i: 16x32 KS2 D4 (M <= 16), 32x32 KS2 D4
# (M ~ 64), 32x64 D4, 64x64 D4, 64x128 D4 (M ~ 128-512).
A4W4_MXFP4_NT_TILE_NAMES = ("256x256", "256x128", "128x256", "128x128", "192x256", "256x192")
A4W4_MXFP4_NT_SMALL_ROWMAJOR_KIDS = tuple(30500 + i for i in range(5))
A4W4_MXFP4_NT_NOSHUFFLE_KIDS = (
    tuple(30100 + i for i in range(6)) + tuple(30110 + i for i in range(6)) + (30130,) + A4W4_MXFP4_NT_SMALL_ROWMAJOR_KIDS
)


def _fake_tune(XQ, WQ, Y, x_scale, w_scale, kernelId) -> Tensor:
    return Y


def _fake_a4w4(XQ, WQ, Y, x_scale, w_scale) -> Tensor:
    return Y


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_a4w4", gen_fake=_fake_a4w4, develop=True)
def _opus_gemm_a4w4_raw(XQ: Tensor, WQ: Tensor, Y: Tensor, x_scale: Tensor, w_scale: Tensor) -> Tensor: ...


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_a4w4_select_kid", develop=True)
def opus_gemm_a4w4_select_kid(M: int, N: int, K: int, batch: int = 1) -> int:
    """Read-only: the kid the opus a4w4 dispatch would run for this shape (tuned table, else heuristic)."""
    ...


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_a4w4_tune", gen_fake=_fake_tune, develop=True)
def _opus_gemm_a4w4_tune_raw(
    XQ: Tensor, WQ: Tensor, Y: Tensor, x_scale: Tensor, w_scale: Tensor, kernelId: int
) -> Tensor: ...


def _as3d(XQ: Tensor, WQ: Tensor, Y: Tensor | None):
    x3 = XQ if XQ.dim() == 3 else XQ.unsqueeze(0)
    w3 = WQ if WQ.dim() == 3 else WQ.unsqueeze(0)
    if Y is None:   # same rank as XQ: 2D input -> [M, N] (as Triton gemm_afp4wfp4), 3D -> [batch, M, N]
        shape = (x3.shape[1], w3.shape[1]) if XQ.dim() == 2 else (x3.shape[0], x3.shape[1], w3.shape[1])
        Y = torch.empty(shape, device=XQ.device, dtype=torch.bfloat16)
    y3 = Y if Y.dim() == 3 else Y.unsqueeze(0)
    return x3, w3, Y, y3


def _check(x3: Tensor, w3: Tensor, x_scale: Tensor, w_scale: Tensor):
    batch, M, K = x3.shape[0], x3.shape[1], x3.shape[2] * 2
    N = w3.shape[1]
    if w3.shape[0] != batch or w3.shape[2] * 2 != K:
        raise ValueError(f"opus a4w4: WQ shape {tuple(w3.shape)} does not match XQ {tuple(x3.shape)}")
    if N % 8:
        raise ValueError(f"opus a4w4: N must be a multiple of 8 (got N={N})")
    if K % 256 or K <= 0:
        raise ValueError(f"opus a4w4: K must be a multiple of 256 (got K={K})")
    for s, rows, nm in ((x_scale, M, "x_scale"), (w_scale, N, "w_scale")):
        if s.dim() not in (2, 3) or (s.dim() == 3 and s.shape[0] != batch) or (s.dim() == 2 and s.shape[0] % batch):
            raise ValueError(f"opus a4w4: {nm} must be 2D [batch*rows, K/32] or [batch, rows, K/32], got {tuple(s.shape)}")
        r = s.shape[1] if s.dim() == 3 else s.shape[0] // batch
        if s.shape[-1] != K // 32 or r < rows:
            raise ValueError(f"opus a4w4: {nm} must be row-major >= {rows} rows x {K // 32} cols per batch, got {tuple(s.shape)}")


def _fp4(t: Tensor) -> Tensor:
    return t.view(aiter.dtypes.fp4x2) if t.dtype == torch.uint8 else t


def gemm_a4w4_opus_noshuffle(
    XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor, Y: Tensor | None = None, kernelId: int | None = None
) -> Tensor:
    """MXFP4 GEMM Y = XQ @ WQ^T with plain row-major inputs (layout of Triton gemm_afp4wfp4): XQ [M, K/2], WQ [N, K/2]
    (fp4x2 or uint8), x_scale [M, K/32], w_scale [N, K/32] (e8m0 or uint8).  kernelId: one of
    A4W4_MXFP4_NT_NOSHUFFLE_KIDS, default None = tuned table, else the C++ heuristic."""
    x3, w3, Y, y3 = _as3d(_fp4(XQ), _fp4(WQ), Y)
    _check(x3, w3, x_scale, w_scale)
    if kernelId is None or kernelId < 0:
        _opus_gemm_a4w4_raw(x3, w3, y3, x_scale, w_scale)
    else:
        if kernelId not in A4W4_MXFP4_NT_NOSHUFFLE_KIDS:
            raise ValueError(f"gemm_a4w4_opus_noshuffle: kernelId {kernelId} is not a row-major kid {A4W4_MXFP4_NT_NOSHUFFLE_KIDS}")
        _opus_gemm_a4w4_tune_raw(x3, w3, y3, x_scale, w_scale, kernelId)
    return Y


__all__ = ["gemm_a4w4_opus_noshuffle", "opus_gemm_a4w4_select_kid", "A4W4_MXFP4_NT_NOSHUFFLE_KIDS"]
