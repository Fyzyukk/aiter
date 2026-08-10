# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Python runtime policy for the OPUS a16w16 family.

Selection is deliberately family-local and ordered as follows:

``explicit kid -> tuned CSV -> architecture heuristic -> framework fallback``

The returned :class:`LaunchConfig` records both the requested kernel and the
launcher that will actually execute.  That distinction is required for the
legacy gfx942 bf16-workspace redirects: workspace planning must use the actual
launcher, never the requested id.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from csrc.opus_gemm.opus_gemm_common import (
    BIAS_AWARE_KIDS,
    GFX942_BF16WS_EXACT_N,
    HEURISTIC_DEFAULT_KIDS_BY_ARCH,
    OpusGemmInstance,
    get_kernel_instance,
    kernel_needs_external_workspace,
)

from .heuristics import A16W16_HEURISTICS
from .heuristics.a16w16_gfx942 import resolve_split_k as resolve_gfx942_split_k

_FAMILY = "a16w16"

TunedLookup = Callable[..., Mapping[str, object] | None]


@dataclass(frozen=True)
class LaunchConfig:
    """Fully resolved a16w16 launch decision.

    ``allocation_split_k`` is a safe allocation upper bound.  On gfx942,
    ``launch_split_k`` is also the already-clamped effective value.  The
    gfx950/gfx1250 launchers retain their existing local down-clamp in Step 1,
    so their launch value remains the caller/CSV request while allocation uses
    ``max(1, request)``.

    A framework fallback has ``requested_kid`` and ``actual_kid`` set to
    ``None``.  Supported OPUS architectures have an always-compiled heuristic
    set; they reach this arm only when the selected raw tune ABI cannot serve
    the request.
    """

    arch: str
    family: str
    source: str
    requested_kid: int | None
    actual_kid: int | None
    requested_split_k: int
    allocation_split_k: int
    launch_split_k: int
    effective_split_k: int | None = None
    fallback_reason: str | None = None

    @property
    def kid(self) -> int | None:
        """Compatibility alias for the launcher id."""
        return self.actual_kid

    @property
    def split_k(self) -> int:
        """Compatibility alias for the value passed to the launcher."""
        return self.launch_split_k

    @property
    def is_framework_fallback(self) -> bool:
        return self.actual_kid is None


def _framework_fallback(arch: str, reason: str) -> LaunchConfig:
    return LaunchConfig(
        arch=arch,
        family=_FAMILY,
        source="framework",
        requested_kid=None,
        actual_kid=None,
        requested_split_k=0,
        allocation_split_k=0,
        launch_split_k=0,
        fallback_reason=reason,
    )


def _output_dtype_name(dtype: Any) -> str:
    value = str(dtype).lower()
    if value in {"bf16", "bfloat16", "bf16_t", "torch.bfloat16"}:
        return "bf16"
    if value in {"fp32", "float", "float32", "fp32_t", "torch.float32"}:
        return "fp32"
    return value


def _instance_output_compatible(
    instance: OpusGemmInstance,
    *,
    needs_workspace: bool,
    output_dtype: Any,
) -> bool:
    # A two-stage kernel always enters tune dispatch as <fp32_t>; its reduce
    # kernel owns the bf16/fp32 final cast.  For a direct-output kernel the
    # instance's existing output_dtypes is authoritative.
    if needs_workspace:
        return _output_dtype_name(output_dtype) in {"bf16", "fp32"}
    token = f"{_output_dtype_name(output_dtype)}_t"
    return token in instance.output_dtypes


def _instance_shape_compatible(
    instance: OpusGemmInstance,
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
) -> bool:
    # The current gfx1250 generated launcher executes exactly one batch.  Keep
    # batched requests out of that raw path until its host loop/ABI is fixed.
    if arch == "gfx1250" and batch != 1:
        return False

    # Mono-tile handles an M tail but has neither an N-tail nor K-tail mask.
    if instance.kernel_tag == "a16w16_mono_tile":
        return N % instance.B_N == 0 and K % instance.B_K == 0

    # Non-OOB mirrors are only valid when their M/N tiles are exact.  K-tail
    # rules remain launcher-owned in Step 1 because they differ by pipeline.
    if not instance.has_oob:
        return M % instance.B_M == 0 and N % instance.B_N == 0
    return True


def _resolve_actual_kid(arch: str, requested_kid: int, N: int) -> int:
    """Resolve legacy gfx942 bf16-workspace host redirects before launch."""
    if arch != "gfx942" or N in GFX942_BF16WS_EXACT_N:
        return requested_kid
    if requested_kid == 10210:
        return 10200
    if requested_kid == 10213:
        return 10203
    if requested_kid == 10216:
        raise ValueError(
            "gfx942 kid 10216 requires exact-N bf16 workspace reduction; "
            f"N={N} is not in {sorted(GFX942_BF16WS_EXACT_N)}"
        )
    return requested_kid


def _build_launch_config(
    *,
    arch: str,
    source: str,
    requested_kid: object,
    requested_split_k: object,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    has_bias: bool,
    output_dtype: Any,
) -> LaunchConfig:
    try:
        requested_kid = int(requested_kid)
        requested_split_k = int(requested_split_k)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid OPUS {source} row: kid={requested_kid!r}, "
            f"splitK={requested_split_k!r}"
        ) from exc

    requested_instance = get_kernel_instance(arch, _FAMILY, requested_kid)
    if requested_instance is None:
        raise ValueError(
            f"OPUS {source} kid {requested_kid} is not an {_FAMILY} kernel "
            f"for {arch}"
        )

    actual_kid = _resolve_actual_kid(arch, requested_kid, N)
    actual_instance = get_kernel_instance(arch, _FAMILY, actual_kid)
    if actual_instance is None:
        raise ValueError(
            f"OPUS requested kid {requested_kid} resolved to unavailable "
            f"{arch}/{_FAMILY} kid {actual_kid}"
        )

    needs_workspace = kernel_needs_external_workspace(arch, _FAMILY, actual_kid)
    if not _instance_shape_compatible(
        actual_instance, arch=arch, M=M, N=N, K=K, batch=batch
    ):
        raise ValueError(
            f"OPUS {source} kid {actual_kid} is incompatible with "
            f"shape (batch={batch}, M={M}, N={N}, K={K})"
        )
    if not _instance_output_compatible(
        actual_instance,
        needs_workspace=needs_workspace,
        output_dtype=output_dtype,
    ):
        raise ValueError(
            f"OPUS {source} kid {actual_kid} does not support output dtype "
            f"{output_dtype}"
        )
    if has_bias and actual_kid not in BIAS_AWARE_KIDS:
        raise ValueError(f"OPUS {source} kid {actual_kid} does not support bias")

    # The generated gfx942 split-K launchers have bias-aware reducers, but the
    # current tune dispatcher intentionally rejects that combination before
    # launcher entry.  Until the raw gate is changed with the workspace ABI,
    # keep automatic selection on the framework fallback.  Explicit callers
    # still receive the strict error from this function.
    if has_bias and arch == "gfx942" and needs_workspace:
        raise ValueError(
            "the current gfx942 tune dispatcher rejects bias on split-K kernels"
        )

    allocation_split_k = 1
    launch_split_k = requested_split_k
    effective_split_k: int | None = None
    if needs_workspace:
        allocation_split_k = max(1, requested_split_k)
        if arch == "gfx942":
            resolution = resolve_gfx942_split_k(
                actual_instance,
                M=M,
                N=N,
                K=K,
                batch=batch,
                cu_num=cu_num,
                requested=requested_split_k,
            )
            allocation_split_k = resolution.allocation
            launch_split_k = resolution.effective
            effective_split_k = resolution.effective

    return LaunchConfig(
        arch=arch,
        family=_FAMILY,
        source=source,
        requested_kid=requested_kid,
        actual_kid=actual_kid,
        requested_split_k=requested_split_k,
        allocation_split_k=allocation_split_k,
        launch_split_k=launch_split_k,
        effective_split_k=effective_split_k,
    )


def select_launch_config(
    *,
    arch: str,
    M: int,
    N: int,
    K: int,
    batch: int,
    cu_num: int,
    has_bias: bool,
    input_dtype: Any,
    output_dtype: Any,
    explicit_kid: int | None = None,
    explicit_split_k: int | None = None,
    tuned_lookup: TunedLookup | None = None,
) -> LaunchConfig:
    """Select and fully resolve an OPUS a16w16 launch.

    Invalid tuned rows are discarded atomically: their kid *and* split-K are
    forgotten before the heuristic runs.  An invalid explicit override is a
    caller error and is therefore reported directly.
    """
    arch = str(arch).lower().split(":", 1)[0]
    M, N, K, batch, cu_num = map(int, (M, N, K, batch, cu_num))
    if min(M, N, K, batch, cu_num) <= 0:
        raise ValueError("M, N, K, batch, and cu_num must all be positive")
    if arch == "gfx1250" and batch != 1:
        raise ValueError(
            "gfx1250 OPUS a16w16 currently requires batch=1; "
            f"got batch={batch}"
        )

    # 1) Explicit override: strict, and never consult the tuned table.
    if explicit_kid is not None:
        return _build_launch_config(
            arch=arch,
            source="explicit",
            requested_kid=explicit_kid,
            requested_split_k=explicit_split_k or 0,
            M=M,
            N=N,
            K=K,
            batch=batch,
            cu_num=cu_num,
            has_bias=bool(has_bias),
            output_dtype=output_dtype,
        )

    # 2) Python tuned CSV.  Load lazily so pure heuristic tests do not pull in
    # pandas, and so monkeypatching common.lookup_tuned remains straightforward.
    if tuned_lookup is None:
        from .common import lookup_tuned

        tuned_lookup = lookup_tuned
    tuned = tuned_lookup(
        M=M,
        N=N,
        K=K,
        bias=bool(has_bias),
        dtype=input_dtype,
        outdtype=output_dtype,
        scaleAB=False,
        bpreshuffle=False,
    )
    if tuned is not None:
        try:
            return _build_launch_config(
                arch=arch,
                source="tuned",
                requested_kid=tuned["solidx"],
                requested_split_k=tuned["splitK"],
                M=M,
                N=N,
                K=K,
                batch=batch,
                cu_num=cu_num,
                has_bias=bool(has_bias),
                output_dtype=output_dtype,
            )
        except (KeyError, TypeError, ValueError):
            # Atomic fallback: do not retain either field from a stale row.
            pass

    # 3) Architecture heuristic.  A returned kid outside the force-compiled
    # set is a source/codegen bug, not a reason to launch some arbitrary kernel.
    heuristic = A16W16_HEURISTICS.get(arch)
    if heuristic is None:
        return _framework_fallback(arch, "no OPUS a16w16 heuristic for this arch")
    requested_kid = int(
        heuristic(
            M,
            N,
            K,
            batch,
            bool(has_bias),
            _output_dtype_name(output_dtype),
        )
    )
    heuristic_kids = HEURISTIC_DEFAULT_KIDS_BY_ARCH.get(arch, frozenset())
    if requested_kid not in heuristic_kids:
        raise RuntimeError(
            f"{arch} a16w16 heuristic returned kid {requested_kid}, which is "
            "not in HEURISTIC_DEFAULT_KIDS_BY_ARCH"
        )
    try:
        return _build_launch_config(
            arch=arch,
            source="heuristic",
            requested_kid=requested_kid,
            requested_split_k=0,
            M=M,
            N=N,
            K=K,
            batch=batch,
            cu_num=cu_num,
            has_bias=bool(has_bias),
            output_dtype=output_dtype,
        )
    except ValueError as exc:
        # 4) The framework owns shapes that the always-available OPUS heuristic
        # cannot legally serve (currently the gfx942 tune ABI's bias gap).
        return _framework_fallback(arch, str(exc))


__all__ = ["LaunchConfig", "select_launch_config"]
