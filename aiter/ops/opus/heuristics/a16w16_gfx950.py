# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Python parity port of the gfx950 a16w16 C++ fallback heuristic."""


def select_kid(
    M: int,
    N: int,
    K: int,
    batch: int = 1,
    has_bias: bool = False,
    output_dtype: str = "bf16",
) -> int:
    """Return the same kid as ``opus_a16w16_heuristic_kid_gfx950``."""
    del batch, output_dtype
    M = int(M)
    N = int(N)
    K = int(K)
    has_bias = bool(has_bias)

    split_barrier_ok = N % 16 == 0 and K % 64 == 0 and (K // 64) % 2 == 0

    if M <= 4:
        if M % 64 == 0 and N % 64 == 0 and K % 128 == 0:
            return 1208
        return 208
    if M <= 64:
        if M % 64 == 0 and N % 32 == 0 and K % 128 == 0:
            return 1206
        return 206
    if M <= 128:
        if M % 64 == 0 and N % 64 == 0 and K % 64 == 0:
            return 1200
        return 200
    if split_barrier_ok and not has_bias:
        if M % 256 == 0 and N % 256 == 0 and K % 64 == 0:
            return 1300
        return 300
    if M % 64 == 0 and N % 64 == 0 and K % 64 == 0:
        return 1200
    return 200


__all__ = ["select_kid"]
