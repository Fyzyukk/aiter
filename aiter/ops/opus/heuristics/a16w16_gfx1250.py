# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 a16w16 fallback selection."""


def select_kid(
    M: int,
    N: int,
    K: int,
    batch: int = 1,
    has_bias: bool = False,
    output_dtype: str = "bf16",
) -> int:
    """Select the gfx1250 fallback kid."""
    del K, batch, has_bias, output_dtype
    M = int(M)
    N = int(N)

    if M % 32 == 0:
        if N % 128 == 0:
            return 20007
        if N % 64 == 0:
            return 20006
        if N % 32 == 0:
            return 20005

    if N % 128 == 0:
        return 20004
    if N % 64 == 0:
        return 20003
    return 20000


__all__ = ["select_kid"]
