# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Architecture-specific a16w16 OPUS fallback heuristics."""

from .a16w16_gfx942 import select_kid as select_kid_gfx942
from .a16w16_gfx950 import select_kid as select_kid_gfx950
from .a16w16_gfx1250 import select_kid as select_kid_gfx1250

A16W16_HEURISTICS = {
    "gfx942": select_kid_gfx942,
    "gfx950": select_kid_gfx950,
    "gfx1250": select_kid_gfx1250,
}


def select_kid(
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int = 1,
    has_bias: bool = False,
    output_dtype: str = "bf16",
) -> int:
    """Dispatch to the per-architecture a16w16 fallback heuristic."""
    arch = str(arch).lower()
    try:
        heuristic = A16W16_HEURISTICS[arch]
    except KeyError as exc:
        raise ValueError(f"unsupported OPUS a16w16 architecture {arch!r}") from exc
    return heuristic(M, N, K, batch, has_bias, output_dtype)


__all__ = [
    "A16W16_HEURISTICS",
    "select_kid",
    "select_kid_gfx942",
    "select_kid_gfx950",
    "select_kid_gfx1250",
]
