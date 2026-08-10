# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Workspace planning for the OPUS a16w16 family."""

from __future__ import annotations

import operator

import torch

from csrc.opus_gemm.opus_gemm_common import (
    OpusGemmInstance,
    get_kernel_instance,
    kernel_needs_external_workspace,
)

from ._workspace import WorkspacePlan, checked_numel

_FAMILY = "a16w16"
_SUPPORTED_ARCHES = frozenset({"gfx942", "gfx950", "gfx1250"})
_WORKSPACE_DTYPES = {
    "bf16_t": torch.bfloat16,
    "fp32_t": torch.float32,
}
_WORKSPACE_ALIGNMENT = 16


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def plan_a16w16_workspace(
    instance: OpusGemmInstance,
    *,
    arch: str,
    kid: int,
    M: int,
    N: int,
    K: int,
    batch: int,
    split_k: int,
) -> WorkspacePlan | None:
    """Build a workspace plan for one fully resolved a16w16 launch.

    ``kid`` must be the actual launcher id, and ``split_k`` must be the
    selector's allocation upper bound rather than its down-clamped launch
    value.  A known non-workspace instance returns ``None``.
    """
    arch = str(arch).lower().split(":", 1)[0]
    if arch not in _SUPPORTED_ARCHES:
        raise ValueError(f"unsupported OPUS a16w16 architecture {arch!r}")
    try:
        kid = operator.index(kid)
    except TypeError as exc:
        raise TypeError(f"kid must be an integer, got {kid!r}") from exc

    canonical = get_kernel_instance(arch, _FAMILY, kid)
    if canonical is None:
        raise KeyError(
            f"unknown OPUS kernel (arch={arch!r}, family={_FAMILY!r}, kid={kid!r})"
        )
    if not isinstance(instance, OpusGemmInstance) or instance is not canonical:
        raise ValueError(
            "a16w16 workspace planning requires the canonical actual "
            f"OpusGemmInstance for ({arch}, {_FAMILY}, {kid})"
        )
    if not kernel_needs_external_workspace(arch, _FAMILY, kid):
        return None

    M = _positive_int(M, name="M")
    N = _positive_int(N, name="N")
    K = _positive_int(K, name="K")
    batch = _positive_int(batch, name="batch")
    split_k = _positive_int(split_k, name="allocation split_k")
    block_m = _positive_int(instance.B_M, name="instance.B_M")
    block_n = _positive_int(instance.B_N, name="instance.B_N")
    block_k = _positive_int(instance.B_K, name="instance.B_K")

    # All current two-stage launchers down-clamp a request until every split
    # owns K work.  More slices than K tiles can never execute, would make the
    # host clamp loop needlessly long, and could request an unbounded tensor.
    max_useful_split_k = _ceil_div(K, block_k)
    if split_k > max_useful_split_k:
        raise ValueError(
            f"allocation split_k={split_k} exceeds the per-kid K-tile limit "
            f"{max_useful_split_k} for K={K}, B_K={block_k}"
        )

    padded_m = _ceil_div(M, block_m) * block_m
    padded_n = _ceil_div(N, block_n) * block_n

    dtype_token = str(instance.splitk_workspace_dtype)
    try:
        dtype = _WORKSPACE_DTYPES[dtype_token]
    except KeyError as exc:
        raise ValueError(
            f"unsupported a16w16 workspace dtype {dtype_token!r} for "
            f"({arch}, {_FAMILY}, {kid})"
        ) from exc
    if arch in {"gfx950", "gfx1250"} and dtype is not torch.float32:
        raise ValueError(
            f"{arch} a16w16 two-stage kernels require fp32 workspace, "
            f"but kid {kid} declares {dtype_token}"
        )

    if arch == "gfx1250":
        if batch != 1:
            raise ValueError(
                "gfx1250 OPUS a16w16 workspace launchers require batch=1; "
                f"got batch={batch}"
            )
        shape = (split_k, padded_m, padded_n)
    else:
        shape = (split_k, batch, padded_m, padded_n)

    required_numel = checked_numel(
        shape,
        name=f"{arch} a16w16 workspace shape",
        limit=(2**63 - 1) // int(dtype.itemsize),
    )
    return WorkspacePlan(
        shape=shape,
        dtype=dtype,
        required_numel=required_numel,
        alignment=_WORKSPACE_ALIGNMENT,
    )


__all__ = ["plan_a16w16_workspace"]
