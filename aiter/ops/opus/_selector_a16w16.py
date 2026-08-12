# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS a16w16 runtime selection.

Resolves ``explicit -> tuned -> heuristic -> fallback`` to the actual kid used
for workspace planning and launch.
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
    """Resolved a16w16 kernel and split-K decision.

    ``allocation_split_k`` sizes the workspace; ``launch_split_k`` is passed
    to the launcher. ``actual_kid=None`` denotes framework fallback.
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
    # Workspace reducers cast output; direct kernels use the registered dtypes.
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
    # The gfx1250 launcher is single-batch.
    if arch == "gfx1250" and batch != 1:
        return False

    if instance.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_fuse":
        split_k = int(instance.fuse_split_k)
        n_cluster = int(instance.fuse_m_cluster)
        if K % 2 != 0 or N % instance.B_N != 0:
            return False
        if split_k < 2 or split_k * n_cluster > 16:
            return False
        num_tiles_n = N // instance.B_N
        if num_tiles_n % n_cluster != 0:
            return False
        return split_k <= (K + instance.B_K - 1) // instance.B_K

    # Mono-tile masks only the M tail.
    if instance.kernel_tag == "a16w16_mono_tile":
        return N % instance.B_N == 0 and K % instance.B_K == 0

    # Non-OOB kernels require exact M/N tiles; launchers validate K.
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
            f"split_k={requested_split_k!r}"
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

    # Fused bias constraints are narrower than the selector can represent.
    if (
        has_bias
        and actual_instance.kernel_tag
        == "a16w16_clusterlaunch_tdm_splitk_fuse"
    ):
        raise ValueError(
            "gfx1250 splitk_fuse has a narrower bf16 [N] bias contract than "
            "the public/tuned selector can represent"
        )

    # The current gfx942 raw ABI rejects workspace kernels with bias.
    if has_bias and arch == "gfx942" and needs_workspace:
        raise ValueError(
            "the current gfx942 a16w16 launch rejects bias on split_k kernels"
        )

    allocation_split_k = 1
    launch_split_k = requested_split_k
    effective_split_k: int | None = None
    if needs_workspace:
        allocation_split_k = max(1, requested_split_k)
        if actual_instance.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_fuse":
            # Fused split-K and workspace size are fixed by the kernel entry.
            effective_split_k = int(actual_instance.fuse_split_k)
            allocation_split_k = effective_split_k
            launch_split_k = effective_split_k
        elif arch == "gfx942":
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
        # cannot legally serve (currently the gfx942 launch ABI's bias gap).
        return _framework_fallback(arch, str(exc))


__all__ = ["LaunchConfig", "select_launch_config"]
