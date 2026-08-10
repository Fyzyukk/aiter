# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Family-neutral helpers for call-scoped Torch workspaces.

The family adapter owns every kernel-specific decision (shape, dtype and
alignment).  This module only represents that decision, allocates the typed
tensor, and applies the validation shared by automatic and caller-provided
workspaces.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import operator
import sys

import torch


def _checked_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    if result <= 0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def checked_numel(
    extents: Iterable[object],
    *,
    name: str = "workspace shape",
    limit: int = sys.maxsize,
) -> int:
    """Multiply positive extents while enforcing a concrete upper bound."""
    limit = _checked_positive_int(limit, name=f"{name} limit")
    values = tuple(extents)
    if not values:
        raise ValueError(f"{name} must contain at least one extent")

    result = 1
    for index, value in enumerate(values):
        extent = _checked_positive_int(value, name=f"{name}[{index}]")
        if result > limit // extent:
            raise OverflowError(
                f"{name} exceeds the supported limit {limit}: extents={values}"
            )
        result *= extent
    return result


@dataclass(frozen=True)
class WorkspacePlan:
    """A concrete, per-call typed workspace allocation contract."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    required_numel: int
    alignment: int

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError(f"workspace dtype must be torch.dtype, got {self.dtype!r}")

        alignment = _checked_positive_int(
            self.alignment, name="workspace alignment"
        )
        if alignment & (alignment - 1):
            raise ValueError(
                f"workspace alignment must be a power of two, got {alignment}"
            )

        itemsize = int(self.dtype.itemsize)
        max_numel = sys.maxsize // itemsize
        allocation_numel = checked_numel(
            shape, name="workspace shape", limit=max_numel
        )
        required_numel = _checked_positive_int(
            self.required_numel, name="workspace required_numel"
        )
        if required_numel > max_numel:
            raise OverflowError(
                "workspace required bytes exceed the supported tensor size: "
                f"required_numel={required_numel}, dtype={self.dtype}"
            )
        if required_numel > allocation_numel:
            raise ValueError(
                "workspace plan shape is smaller than required_numel: "
                f"shape={shape} ({allocation_numel} elements), "
                f"required_numel={required_numel}"
            )

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "required_numel", required_numel)
        object.__setattr__(self, "alignment", alignment)

    @property
    def allocation_numel(self) -> int:
        """Number of elements allocated by :func:`allocate_workspace`."""
        return checked_numel(
            self.shape,
            name="workspace shape",
            limit=sys.maxsize // int(self.dtype.itemsize),
        )


def validate_workspace(
    workspace: torch.Tensor,
    plan: WorkspacePlan,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Validate a workspace tensor and return it unchanged.

    Shape is intentionally not prescribed for caller-owned buffers.  Kernels
    consume a contiguous address range, so a flat or differently shaped tensor
    is valid when its dtype, device, alignment, and element capacity satisfy
    the plan.
    """
    if not isinstance(plan, WorkspacePlan):
        raise TypeError(f"plan must be WorkspacePlan, got {type(plan).__name__}")
    if not isinstance(workspace, torch.Tensor):
        raise TypeError(
            f"workspace must be a torch.Tensor, got {type(workspace).__name__}"
        )

    if workspace.dtype != plan.dtype:
        raise ValueError(
            f"workspace dtype mismatch: expected {plan.dtype}, got {workspace.dtype}"
        )
    if device is not None:
        expected_device = torch.device(device)
        if workspace.device != expected_device:
            raise ValueError(
                "workspace device mismatch: "
                f"expected {expected_device}, got {workspace.device}"
            )
    if not workspace.is_contiguous():
        raise ValueError(
            "workspace must be contiguous; "
            f"got shape={tuple(workspace.shape)}, stride={tuple(workspace.stride())}"
        )
    if workspace.numel() < plan.required_numel:
        raise ValueError(
            "workspace capacity is too small: "
            f"need at least {plan.required_numel} elements, got {workspace.numel()}"
        )

    try:
        data_ptr = int(workspace.data_ptr())
    except RuntimeError as exc:
        raise ValueError("workspace must have addressable storage") from exc
    if data_ptr == 0 or data_ptr % plan.alignment != 0:
        raise ValueError(
            "workspace address is not sufficiently aligned: "
            f"data_ptr=0x{data_ptr:x}, required_alignment={plan.alignment}"
        )
    return workspace


def allocate_workspace(
    plan: WorkspacePlan,
    device: torch.device | str | int,
) -> torch.Tensor:
    """Allocate and validate one call-scoped typed workspace tensor.

    No tensor is cached here.  Reuse and graph-private ownership are delegated
    to PyTorch's device caching allocator.
    """
    if not isinstance(plan, WorkspacePlan):
        raise TypeError(f"plan must be WorkspacePlan, got {type(plan).__name__}")
    target_device = torch.device(device)
    workspace = torch.empty(plan.shape, dtype=plan.dtype, device=target_device)
    return validate_workspace(workspace, plan, target_device)


__all__ = [
    "WorkspacePlan",
    "allocate_workspace",
    "checked_numel",
    "validate_workspace",
]
