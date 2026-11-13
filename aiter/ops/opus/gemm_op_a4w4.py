# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus gfx950 MXFP4 x MXFP4 NT GEMM (a4w4_mxfp4_nt, 1x32 e8m0 scales).

Inputs follow aiter ``gemm_a4w4(..., bpreshuffle=True)``:
  * ``XQ``: [M, K/2] (or [batch, M, K/2]) fp4x2, row-major.
  * ``WQ``: [N, K/2] (or [batch, N, K/2]) fp4x2 in ``shuffle_weight(w, layout=(16, 16))`` layout.
  * ``x_scale`` / ``w_scale``: e8m0 in ``e8m0_shuffle`` layout ([rows padded to 256 (32 is enough), K/32 padded to 8]
    per batch; for batch > 1 either [batch, rows, cols] or [batch * rows, cols]).
  * output bf16 [M, N] (or [batch, M, N]).
Any M; N % 16 == 0; K % 256 == 0. Device code is the pre-compiled gen_co/gfx950/*.co (pin toolchain).
"""

import torch
from torch import Tensor

from ...jit.core import compile_ops

A4W4_MXFP4_NT_KIDS = (30000, 30001, 30002, 30003)   # tiles 256x256, 256x128, 128x256, 128x128


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


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_a4w4_tune", gen_fake=_fake_tune, develop=True)
def _opus_gemm_a4w4_tune_raw(
    XQ: Tensor, WQ: Tensor, Y: Tensor, x_scale: Tensor, w_scale: Tensor, kernelId: int
) -> Tensor: ...


def _as3d(XQ: Tensor, WQ: Tensor, Y: Tensor | None):
    x3 = XQ if XQ.dim() == 3 else XQ.unsqueeze(0)
    w3 = WQ if WQ.dim() == 3 else WQ.unsqueeze(0)
    if Y is None:
        Y = torch.empty((x3.shape[0], x3.shape[1], w3.shape[1]), device=XQ.device, dtype=torch.bfloat16)
    y3 = Y if Y.dim() == 3 else Y.unsqueeze(0)
    return x3, w3, Y, y3


def gemm_a4w4_opus(XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor, Y: Tensor | None = None) -> Tensor:
    """MXFP4 GEMM Y = XQ @ WQ^T with 1x32 scales; tile chosen by the tuned table, else the C++ heuristic."""
    x3, w3, Y, y3 = _as3d(XQ, WQ, Y)
    _opus_gemm_raw(x3, w3, y3, None, x_scale, w_scale, None)
    return Y


def opus_gemm_a4w4_tune(
    XQ: Tensor, WQ: Tensor, x_scale: Tensor, w_scale: Tensor, Y: Tensor | None = None, kernelId: int = 30000
) -> Tensor:
    """Run one a4w4_mxfp4_nt kid (30000..30003) explicitly."""
    x3, w3, Y, y3 = _as3d(XQ, WQ, Y)
    _opus_gemm_a4w4_tune_raw(x3, w3, y3, x_scale, w_scale, kernelId)
    return Y


__all__ = ["gemm_a4w4_opus", "opus_gemm_a4w4_tune", "A4W4_MXFP4_NT_KIDS"]
