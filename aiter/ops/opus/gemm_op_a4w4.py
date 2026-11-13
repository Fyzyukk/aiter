# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus gfx950 MXFP4 x MXFP4 NT GEMM (1x32 e8m0 scales), two input layouts:

``gemm_a4w4_opus`` (kernel a4w4_mxfp4_nt, kids 300xx) -- same inputs as aiter ``gemm_a4w4(..., bpreshuffle=True)``; normally
reached through ``aiter.gemm_a4w4`` (tuned-CSV kernelName ``opus_a4w4_kid<kid>`` / ``opus_a4w4_auto``, or
``AITER_A4W4_BACKEND=opus``):
  * ``XQ``: [M, K/2] (or [batch, M, K/2]) fp4x2, row-major.
  * ``WQ``: [N, K/2] fp4x2 in ``shuffle_weight(w, layout=(16, 16))`` layout.
  * ``x_scale`` / ``w_scale``: e8m0 in ``e8m0_shuffle`` layout ([rows padded to 32 or 256, K/32 padded to 8] per batch).
  Any M; N % 16 == 0; K % 256 == 0.

``gemm_a4w4_opus_noshuffle`` (kernel a4w4_mxfp4_nt_noshuffle, kids 301xx) -- plain row-major inputs, the layout of Triton
``gemm_afp4wfp4``: ``XQ`` [M, K/2], ``WQ`` [N, K/2] (fp4x2 or uint8), ``x_scale`` [M, K/32], ``w_scale`` [N, K/32]
(e8m0 or uint8).  Any M; N % 8 == 0; K % 256 == 0.

Output bf16 [M, N] (or [batch, M, N]).  Device code is the pre-compiled gen_co/gfx950/*.co (pin toolchain).
"""

import torch
from torch import Tensor

import aiter

from ...jit.core import compile_ops

# kid = 30000 + variant offset + tile index; tiles 256x256, 256x128, 128x256, 128x128, 192x256, 256x192.
# variants: +0 persistent, +10 one tile per block, +20 split-K (no 256x256; S from the C++ heuristic or OPUS_MXFP4_SPLITK),
# 30030 = persistent 256x256 without M/N boundary handling (M, N multiples of 256).
A4W4_MXFP4_NT_TILE_NAMES = ("256x256", "256x128", "128x256", "128x128", "192x256", "256x192")
A4W4_MXFP4_NT_KIDS = (
    tuple(30000 + i for i in range(6)) + tuple(30010 + i for i in range(6)) + tuple(30021 + i for i in range(5)) + (30030,)
)
# Small-tile pipeline (JIT-built), shuffle layout: 30400..30408 (B_DIRECT 16x32 KS4 D3/D4, 32x32 KS4 D2, 32x64 KS4 D2 / KS2 D3,
# 16x64 KS4 D3; B_LDS 32x128 D3/D4, 64x128 D4); the opus dispatch uses them for M <= 64.  30410..30413: A_LDS (A in LDS, B direct,
# asm-style) 32x64 D6, 32x128 D3/D4, 64x128 D4 for M ~ 128-512 (tuner candidates).
A4W4_MXFP4_NT_SMALL_KIDS = tuple(30400 + i for i in range(9)) + tuple(30410 + i for i in range(4))
# Row-major kids: 30100 + i persistent, 30110 + i one tile per block, 30130 = 256x256 without boundary; small-tile
# 30500 + 3 * i + {0, 1, 2} = split 1 / 2 / 4 of (B_LDS 16x64 D3/D4, 32x64 D4, 64x64 D3, 64x128 D3/D4; B_DIRECT 16x32 KS4 D2).
A4W4_MXFP4_NT_SMALL_ROWMAJOR_KIDS = tuple(30500 + i for i in range(21))
A4W4_MXFP4_NT_NOSHUFFLE_KIDS = (
    tuple(30100 + i for i in range(6)) + tuple(30110 + i for i in range(6)) + (30130,) + A4W4_MXFP4_NT_SMALL_ROWMAJOR_KIDS
)


def _fake_dispatch(XQ, WQ, Y, group_layout=None, x_scale=None, w_scale=None, bias=None) -> Tensor:
    return Y


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm", gen_fake=_fake_dispatch, develop=True)
def _opus_gemm_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Tensor | None = None,
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
    bias: Tensor | None = None,
) -> Tensor: ...


def _fake_tune(XQ, WQ, Y, x_scale, w_scale, kernelId) -> Tensor:
    return Y


def _fake_a4w4(XQ, WQ, Y, x_scale, w_scale, bpreshuffle=True) -> Tensor:
    return Y


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_a4w4", gen_fake=_fake_a4w4, develop=True)
def _opus_gemm_a4w4_raw(
    XQ: Tensor, WQ: Tensor, Y: Tensor, x_scale: Tensor, w_scale: Tensor, bpreshuffle: bool = True
) -> Tensor: ...


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_a4w4_select_kid", develop=True)
def opus_gemm_a4w4_select_kid(M: int, N: int, K: int, batch: int = 1, noshuffle: bool = False) -> int:
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


def _check(x3: Tensor, w3: Tensor, x_scale: Tensor, w_scale: Tensor, bpreshuffle: bool = True):
    batch, M, K = x3.shape[0], x3.shape[1], x3.shape[2] * 2
    N = w3.shape[1]
    if w3.shape[0] != batch or w3.shape[2] * 2 != K:
        raise ValueError(f"opus a4w4: WQ shape {tuple(w3.shape)} does not match XQ {tuple(x3.shape)}")
    if not bpreshuffle:
        if N % 8:
            raise ValueError(f"opus a4w4 (bpreshuffle=False): N must be a multiple of 8 (got N={N})")
        if K % 256 or K <= 0:
            raise ValueError(f"opus a4w4: K must be a multiple of 256 (got K={K})")
        for s, rows, nm in ((x_scale, M, "x_scale"), (w_scale, N, "w_scale")):
            if s.dim() not in (2, 3) or (s.dim() == 3 and s.shape[0] != batch) or (s.dim() == 2 and s.shape[0] % batch):
                raise ValueError(f"opus a4w4: {nm} must be 2D [batch*rows, K/32] or [batch, rows, K/32], got {tuple(s.shape)}")
            r = s.shape[1] if s.dim() == 3 else s.shape[0] // batch
            if s.shape[-1] != K // 32 or r < rows:
                raise ValueError(f"opus a4w4 (bpreshuffle=False): {nm} must be row-major >= {rows} rows x {K // 32} cols per batch, got {tuple(s.shape)}")
        return
    if N % 16:
        raise ValueError(f"opus a4w4: N must be a multiple of 16 (got N={N})")
    if K % 256 or K <= 0:
        raise ValueError(f"opus a4w4: K must be a multiple of 256 (got K={K})")
    cols = (K // 32 + 7) // 8 * 8
    for s, rows, nm in ((x_scale, M, "x_scale"), (w_scale, N, "w_scale")):
        if s.dim() not in (2, 3) or (s.dim() == 3 and s.shape[0] != batch) or (s.dim() == 2 and s.shape[0] % batch):
            raise ValueError(f"opus a4w4: {nm} must be 2D [batch*rows, cols] or [batch, rows, cols], got {tuple(s.shape)}")
        r = s.shape[1] if s.dim() == 3 else s.shape[0] // batch
        need = (rows + 31) // 32 * 32
        if s.shape[-1] != cols or r < need or r % 32:
            raise ValueError(f"opus a4w4: {nm} must be e8m0_shuffle'd: >= {need} rows (multiple of 32) x {cols} cols per batch, got {tuple(s.shape)}")


def gemm_a4w4_opus(XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor, Y: Tensor | None = None) -> Tensor:
    """MXFP4 GEMM Y = XQ @ WQ^T, inputs as aiter gemm_a4w4(bpreshuffle=True) (shuffle_weight B, e8m0_shuffle scales);
    tile chosen by the tuned table, else the C++ heuristic."""
    x3, w3, Y, y3 = _as3d(XQ, WQ, Y)
    _check(x3, w3, x_scale, w_scale, True)
    _opus_gemm_a4w4_raw(x3, w3, y3, x_scale, w_scale, True)
    return Y


def _fp4(t: Tensor) -> Tensor:
    return t.view(aiter.dtypes.fp4x2) if t.dtype == torch.uint8 else t


def gemm_a4w4_opus_noshuffle(
    XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor, Y: Tensor | None = None, kernelId: int | None = None
) -> Tensor:
    """MXFP4 GEMM Y = XQ @ WQ^T with plain row-major inputs (layout of Triton gemm_afp4wfp4): XQ [M, K/2], WQ [N, K/2]
    (fp4x2 or uint8), x_scale [M, K/32], w_scale [N, K/32] (e8m0 or uint8).  kernelId: one of
    A4W4_MXFP4_NT_NOSHUFFLE_KIDS, default None = tuned table, else the C++ heuristic."""
    x3, w3, Y, y3 = _as3d(_fp4(XQ), _fp4(WQ), Y)
    _check(x3, w3, x_scale, w_scale, False)
    if kernelId is None or kernelId < 0:
        _opus_gemm_a4w4_raw(x3, w3, y3, x_scale, w_scale, False)
    else:
        if kernelId not in A4W4_MXFP4_NT_NOSHUFFLE_KIDS:
            raise ValueError(f"gemm_a4w4_opus_noshuffle: kernelId {kernelId} is not a row-major kid {A4W4_MXFP4_NT_NOSHUFFLE_KIDS}")
        _opus_gemm_a4w4_tune_raw(x3, w3, y3, x_scale, w_scale, kernelId)
    return Y


def opus_gemm_a4w4_tune(
    XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor, Y: Tensor | None = None, kernelId: int = 30000
) -> Tensor:
    """Run one kid explicitly: A4W4_MXFP4_NT_KIDS (bpreshuffle layout) or A4W4_MXFP4_NT_NOSHUFFLE_KIDS (row-major)."""
    x3, w3, Y, y3 = _as3d(XQ, WQ, Y)
    _check(x3, w3, x_scale, w_scale, kernelId not in A4W4_MXFP4_NT_NOSHUFFLE_KIDS)
    _opus_gemm_a4w4_tune_raw(x3, w3, y3, x_scale, w_scale, kernelId)
    return Y


__all__ = ["gemm_a4w4_opus", "opus_gemm_a4w4_select_kid", "gemm_a4w4_opus_noshuffle", "opus_gemm_a4w4_tune", "A4W4_MXFP4_NT_KIDS", "A4W4_MXFP4_NT_SMALL_KIDS",
           "A4W4_MXFP4_NT_NOSHUFFLE_KIDS"]
