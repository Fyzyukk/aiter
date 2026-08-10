# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 a16w16 fallback heuristic and split-K resolver.

The C++ heuristic still returns launcher function pointers.  This parity port
keeps the branch decisions expressed in launcher-symbol names, then resolves
those names through the canonical ``OpusGemmInstance.name`` values.  No second
hand-written symbol-to-kid table is maintained.
"""

from dataclasses import dataclass

from csrc.opus_gemm.opus_gemm_common import (
    HEURISTIC_DEFAULT_KIDS_GFX942,
    OpusGemmInstance,
    get_kernel_instance,
)

_FAMILY = "a16w16"
_MAX_AUTO_SPLIT_K = 16
_MIN_ITERS_PER_SPLIT = 2
_EVEN_LOOP_SPLITK_TAGS = frozenset(
    {
        "a16w16_kbuf2v_sk",
        "a16w16_kbuf2v_bk128_sk",
        "a16w16_quad_mfma32_kbuf1_sk",
    }
)


def _build_symbol_to_kid() -> dict[str, int]:
    result: dict[str, int] = {}
    for kid in HEURISTIC_DEFAULT_KIDS_GFX942:
        instance = get_kernel_instance("gfx942", _FAMILY, kid)
        if instance is None:
            raise RuntimeError(f"gfx942 heuristic kid {kid} has no a16w16 instance")
        previous = result.setdefault(instance.name, kid)
        if previous != kid:
            raise RuntimeError(
                f"duplicate gfx942 launcher symbol {instance.name!r}: "
                f"kids {previous} and {kid}"
            )
    return result


_SYMBOL_TO_KID = _build_symbol_to_kid()


def launcher_symbol_to_kid(symbol: str) -> int:
    """Resolve a generated launcher symbol through canonical instance names."""
    try:
        return _SYMBOL_TO_KID[symbol]
    except KeyError as exc:
        raise RuntimeError(
            f"gfx942 heuristic returned unknown launcher symbol {symbol!r}"
        ) from exc


def _split_barrier_ok(N: int, K: int) -> bool:
    loops = (K + 63) // 64
    return N % 16 == 0 and K % 64 == 0 and loops >= 2 and loops % 2 == 0


def _bf16ws_band(M: int, N: int, K: int) -> bool:
    return (
        K >= 4096
        and K % 64 == 0
        and 104 <= M <= 608
        and (N == 256 or 512 <= N <= 2048)
    )


def _select_bf16_symbol(M: int, N: int, K: int) -> str:
    k64_ok = K % 64 == 0
    k32_ok = K % 32 == 0
    wkc_bk64_ok = K >= 4096 and K % 512 == 0
    p1_ok = K % 128 == 0
    sb_ok = _split_barrier_ok(N, K)

    if K == 4096:
        if p1_ok and (M in (48, 64) and N == 1024):
            return "opus_gemm_gfx942_splitk_p1_bk128_bf16ws_256x64x64x128_2x2_16x16x16_0x0x0"
        if p1_ok and ((M == 128 and N == 512) or (M == 256 and N == 256)):
            return "opus_gemm_gfx942_splitk_p1_bk128_bf16ws_256x64x64x128_2x2_16x16x16_0x0x0"
        if p1_ok and M == 512 and N == 256:
            return "opus_gemm_gfx942_splitk_p1_bk128_256x64x64x128_2x2_16x16x16_0x0x0"
        if M in (48, 64) and 1536 <= N <= 2048:
            return "opus_gemm_gfx942_splitk_legacy_512x64x128x64_2x4_16x16x16_0x0x0"
        if (M == 128 and N == 1024) or (M == 256 and N == 512):
            return "opus_gemm_gfx942_splitk_legacy_512x64x128x64_2x4_16x16x16_0x0x0"
        if (
            (M == 128 and 1536 <= N <= 2048)
            or (M == 256 and N == 1024)
            or (M == 512 and N == 512)
        ):
            return "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"

    if K >= 1024 and k32_ok and N >= 1536 and M <= 32:
        if M <= 4 and N >= 4096:
            return "opus_gemm_gfx942_wkc_512x16x16x64_1x1_16x16x16_0x0x0"
        if M <= 16:
            if wkc_bk64_ok:
                return "opus_gemm_gfx942_wkc_512x16x32x64_1x1_16x16x16_0x0x0"
            return "opus_gemm_gfx942_wkc_512x16x32x32_1x1_16x16x16_0x0x0"
        if M == 32 and K == 4096 and wkc_bk64_ok:
            return "opus_gemm_gfx942_wkc_512x16x32x64_1x1_16x16x16_0x0x0"
        return "opus_gemm_gfx942_wkc_256x32x32x64_1x1_16x16x16_0x0x0"

    if K >= 512 and k64_ok and (
        N <= 64 or (M <= 128 and N <= 1024) or (M <= 8 and N <= 1536)
    ):
        if N <= 64 and M > 128:
            return "opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0"
        if N <= 256 or M <= 8 or (M <= 16 and N <= 800):
            return "opus_gemm_gfx942_wkc_512x16x16x64_1x1_16x16x16_0x0x0"
        return "opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0"

    if _bf16ws_band(M, N, K):
        return "opus_gemm_gfx942_splitk_legacy_bf16ws_512x128x128x64_2x4_16x16x16_0x0x0"

    if N == 384 and K >= 4096:
        if M <= 128:
            return "opus_gemm_gfx942_wkc_512x32x16x64_1x1_16x16x16_0x0x0"
        if M <= 224:
            return "opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0"
        if 392 <= M <= 512:
            return "opus_gemm_gfx942_splitk_em3en4_lds1_pgr2_256x128x96x128_2x2_16x16x16_0x0x0"
        return "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"

    if k64_ok and N >= 4096 and K <= 3200:
        if K <= 640 and M <= 128:
            return "opus_gemm_gfx942_p1_256x64x64x64_2x2_16x16x16_0x0x0"
        return "opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0"

    if sb_ok and M >= 128:
        return "opus_gemm_gfx942_512x128x128x64_2x4_16x16x16_0x0x0"
    if N <= 256 and p1_ok:
        return "opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0"
    return "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"


def select_kid(
    M: int,
    N: int,
    K: int,
    batch: int = 1,
    has_bias: bool = False,
    output_dtype: str = "bf16",
) -> int:
    """Return the gfx942 C++ heuristic's launcher as a canonical kid."""
    del batch
    M = int(M)
    N = int(N)
    K = int(K)
    output_dtype = str(output_dtype).lower()

    if output_dtype in {"bf16", "bfloat16", "bf16_t", "torch.bfloat16"} and not has_bias:
        return launcher_symbol_to_kid(_select_bf16_symbol(M, N, K))
    if N <= 256 and K % 128 == 0:
        return launcher_symbol_to_kid(
            "opus_gemm_gfx942_splitk_p1_256x64x64x64_2x2_16x16x16_0x0x0"
        )
    return launcher_symbol_to_kid(
        "opus_gemm_gfx942_splitk_legacy_512x128x128x64_2x4_16x16x16_0x0x0"
    )


@dataclass(frozen=True)
class SplitKResolution:
    """gfx942 split-K request, allocation upper bound, and clamped value."""

    requested: int
    allocation: int
    effective: int


def resolve_split_k(
    instance: OpusGemmInstance,
    *,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    requested: int,
) -> SplitKResolution:
    """Mirror the generated gfx942 launcher's auto-pick and down-clamp."""
    M = int(M)
    N = int(N)
    K = int(K)
    batch = int(batch)
    cu_num = int(cu_num)
    requested = int(requested)
    if min(M, N, K, batch, cu_num) <= 0:
        raise ValueError("M, N, K, batch, and cu_num must all be positive")

    if requested > 0:
        allocation = requested
    else:
        tiles_mn = (
            (M + instance.B_M - 1)
            // instance.B_M
            * ((N + instance.B_N - 1) // instance.B_N)
            * batch
        )
        tiles_mn = max(1, tiles_mn)
        # Preserve the current generated launcher predicate verbatim for the
        # Step-1 parity commit.  Current kernel tags do not end in ``_p1``;
        # changing the intended P1 multiplier belongs in a separately paired
        # Python/C++ behavior change, not this mechanical migration.
        target_wg = (2 * cu_num) if instance.kernel_tag.endswith("_p1") else cu_num
        allocation = (target_wg + tiles_mn - 1) // tiles_mn
        allocation = min(_MAX_AUTO_SPLIT_K, max(1, allocation))

    total_iters = (K + instance.B_K - 1) // instance.B_K
    if total_iters < _MIN_ITERS_PER_SPLIT:
        raise ValueError(
            f"K={K} is too small for gfx942 kid B_K={instance.B_K}; "
            f"need at least {instance.B_K * _MIN_ITERS_PER_SPLIT}"
        )

    effective = allocation
    require_even = instance.kernel_tag in _EVEN_LOOP_SPLITK_TAGS
    while effective > 1:
        iters_full = (total_iters + effective - 1) // effective
        last_loops = total_iters - (effective - 1) * iters_full
        parity_ok = not require_even or (
            iters_full % 2 == 0 and last_loops % 2 == 0
        )
        if (
            iters_full >= _MIN_ITERS_PER_SPLIT
            and last_loops >= _MIN_ITERS_PER_SPLIT
            and parity_ok
        ):
            break
        effective -= 1

    if require_even:
        iters_full = (total_iters + effective - 1) // effective
        last_loops = total_iters - (effective - 1) * iters_full
        if iters_full % 2 != 0 or last_loops % 2 != 0:
            raise ValueError(
                f"gfx942 kid {instance.name} needs even loops per split; "
                f"K={K}, split_k={effective}, loops=({iters_full},{last_loops})"
            )

    return SplitKResolution(
        requested=requested,
        allocation=allocation,
        effective=effective,
    )


__all__ = [
    "SplitKResolution",
    "launcher_symbol_to_kid",
    "resolve_split_k",
    "select_kid",
]
