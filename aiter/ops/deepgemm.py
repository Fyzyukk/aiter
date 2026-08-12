# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepGEMM CK binding and deprecated OPUS compatibility entry."""

import warnings

import torch
from torch import Tensor

from ..jit.core import compile_ops
from .opus.gemm_op_a16w16 import opus_gemm_a16w16_launch as _opus_launch


@compile_ops("module_deepgemm", fc_name="deepgemm")
def deepgemm_ck(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Tensor,
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
) -> Tensor: ...


def deepgemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Tensor,
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
):
    return deepgemm_ck(XQ, WQ, Y, group_layout, x_scale, w_scale)


def opus_gemm_a16w16_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    """Forward the legacy tune name to the OPUS exact-kid launcher."""
    warnings.warn(
        "aiter.ops.deepgemm.opus_gemm_a16w16_tune has moved to "
        "aiter.ops.opus.gemm_op_a16w16.opus_gemm_a16w16_launch; this "
        "shim will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _opus_launch(XQ, WQ, Y, kid=kernelId, split_k=splitK)


__all__ = [
    "deepgemm",
    "deepgemm_ck",
    "opus_gemm_a16w16_tune",
]
