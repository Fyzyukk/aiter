# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public OPUS GEMM/BMM interfaces backed by shared exact-kid launchers."""

from __future__ import annotations

import torch
from torch import Tensor

from .dispatch import _opus_dispatch


def opus_gemm_bpreshuffle(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    *,
    pipeline: str,
    config=None,
    split_k: int = 0,
    workspace: Tensor | None = None,
    **compile_params,
) -> Tensor:
    """Launch native-E8M0 GEMM by pipeline and static compile parameters.

    The five pipelines are ``pin``, ``tiled``, ``register``, ``lds`` and
    ``large_output``. ``config`` accepts an enumerated BpreshuffleConfig, its
    complete JSON payload, or its parameter mapping. Named compile parameters
    can instead identify one registered configuration. Tile, wave and queue
    parameters select compiled specializations; runtime split-K is a separate
    launch argument. The internal compatibility ID is resolved automatically.
    """
    from csrc.opus_gemm.opus_gemm_bpreshuffle_config import resolve_config

    selected = resolve_config(pipeline, config, **compile_params)
    return _opus_dispatch(
        "opus_gemm",
        2,
        XQ,
        WQ,
        Y,
        kid=selected.legacy_kid,
        layout="bpreshuffle",
        x_scale=x_scale,
        w_scale=w_scale,
        split_k=split_k,
        workspace=workspace,
    )


def opus_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    layout: str = "plain",
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
    bias: Tensor | None = None,
    split_k: int = 0,
    workspace: Tensor | None = None,
) -> Tensor:
    """Launch logical 2D ``[M,K] x [N,K] -> [M,N]`` by exact ``kid``.

    ``Y`` is caller-owned and returned. ``layout='bpreshuffle'`` declares a
    transformed WQ content layout that Tensor metadata cannot prove.

    ``kid`` is mandatory and selects the exact registered kernel configuration;
    this entry does not choose a kid from the shape. For gfx950 native-E8M0
    bpreshuffle runtime IDs 92310/92311/92320/92321/92330/92340 and
    92410/92420/92430, ``split_k`` selects a launch parameter within that kid:
    positive values are literal counts in ``1..min(16, K/128)``, zero keeps the
    historical default (register four, fine one), and minus one requests the
    optional shape/CU grid heuristic. The heuristic chooses only the split
    count and has no measured performance guarantee. Fixed-split bpreshuffle
    IDs retain ``split_k=0``. A split greater than one uses a call-scoped FP32
    workspace; split one rejects a supplied workspace.
    """
    return _opus_dispatch(
        "opus_gemm",
        2,
        XQ,
        WQ,
        Y,
        kid=kid,
        layout=layout,
        x_scale=x_scale,
        w_scale=w_scale,
        bias=bias,
        split_k=split_k,
        workspace=workspace,
    )


def opus_bmm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int,
    layout: str = "plain",
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
    bias: Tensor | None = None,
    split_k: int = 0,
    workspace: Tensor | None = None,
) -> Tensor:
    """Launch batch-first ``[B,M,K] x [B,N,K] -> [B,M,N]`` by exact kid."""
    return _opus_dispatch(
        "opus_bmm",
        3,
        XQ,
        WQ,
        Y,
        kid=kid,
        layout=layout,
        x_scale=x_scale,
        w_scale=w_scale,
        bias=bias,
        split_k=split_k,
        workspace=workspace,
    )


def gemm_a16w16_opus(
    A: Tensor,
    B: Tensor,
    bias: Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    *,
    kernelId: int | None = None,
    splitK: int | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Run the legacy shape-driven A16W16 OPUS selection path."""
    from .gemm_op_a16w16 import gemm_a16w16_opus as _impl

    return _impl(
        A,
        B,
        bias,
        dtype,
        kernelId=kernelId,
        splitK=splitK,
        out=out,
    )


__all__ = [
    "gemm_a16w16_opus",
    "opus_bmm",
    "opus_gemm",
    "opus_gemm_bpreshuffle",
]
