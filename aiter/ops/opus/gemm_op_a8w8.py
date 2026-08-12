# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Canonical OPUS A8W8 family launch APIs and legacy compatibility shim.

Each public wrapper resolves an exact ``(runtime arch, logical family, kid,
Y.dtype)`` registry entry before calling its private raw binding. Physical
shape, stride, tile, scale and prefetch contracts remain final-checked by the
generated launcher for that exact instance.
"""

from __future__ import annotations

import functools
import warnings

import torch
from torch import Tensor

from csrc.opus_gemm.opus_gemm_common import get_kernel_instance

from ...jit.core import compile_ops

_A8W8_FAMILY = "a8w8"
_A8W8_BLOCKSCALE_FAMILY = "a8w8_blockscale"
_A8W8_BPRESHUFFLE_FAMILY = "a8w8_blockscale_bpreshuffle"
_SUPPORTED_OPUS_ARCHES = ("gfx942", "gfx950", "gfx1250")
_MISSING_TENSOR = object()

# A tensor's explicit device index is stable for the process lifetime, as is
# the architecture attached to that visible device.  Keep this cache scoped by
# the full torch.device instead of assuming a homogeneous multi-GPU host.
_DEVICE_ARCH_CACHE: dict[torch.device, str] = {}

# A default is a deliberate no-tuned-row fallback, never "the first registry
# entry".  Empty capabilities stay None until that arch lands a real kernel.
OPUS_DEFAULT_A8W8_BPRESHUFFLE_KID_BY_ARCH: dict[str, int | None] = {
    "gfx942": 11000,
    "gfx950": None,
    "gfx1250": None,
}


def _read_device_arch(device: torch.device) -> str:
    """Read one explicit device's gfx name from the runtime."""
    props = torch.cuda.get_device_properties(device)
    raw_arch = str(getattr(props, "gcnArchName", "")).strip()
    arch = raw_arch.split(":", 1)[0].lower()
    if not arch.startswith("gfx"):
        try:
            from ...jit.utils.chip_info import get_gfx_runtime

            arch = get_gfx_runtime().lower()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"cannot determine the AMD gfx architecture for device {device}"
            ) from exc
    return arch


def _device_arch(device: torch.device) -> str:
    """Return the tensor device's cached gfx name.

    ``Tensor.device`` is explicit, but normalizing an index-less ``cuda``
    device keeps this helper correct for direct internal callers that switch
    the current device.  The cache never stores Tensor objects or data
    pointers.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    arch = _DEVICE_ARCH_CACHE.get(device)
    if arch is None:
        arch = _read_device_arch(device)
        _DEVICE_ARCH_CACHE[device] = arch
    return arch


def _check_same_device(
    entry: str,
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: object = _MISSING_TENSOR,
    w_scale: object = _MISSING_TENSOR,
) -> None:
    """Validate dynamic Tensor devices without allocating on valid calls."""
    has_x_scale = x_scale is not _MISSING_TENSOR
    has_w_scale = w_scale is not _MISSING_TENSOR
    if not (
        isinstance(XQ, Tensor)
        and isinstance(WQ, Tensor)
        and isinstance(Y, Tensor)
        and (not has_x_scale or isinstance(x_scale, Tensor))
        and (not has_w_scale or isinstance(w_scale, Tensor))
    ):
        named_tensors = [("XQ", XQ), ("WQ", WQ), ("Y", Y)]
        if has_x_scale:
            named_tensors.append(("x_scale", x_scale))
        if has_w_scale:
            named_tensors.append(("w_scale", w_scale))
        invalid = [
            name for name, tensor in named_tensors if not isinstance(tensor, Tensor)
        ]
        raise TypeError(
            f"{entry}: {', '.join(invalid)} must be Tensor objects"
        )

    device = XQ.device
    same_device = WQ.device == device and Y.device == device
    if has_x_scale:
        same_device = same_device and x_scale.device == device
    if has_w_scale:
        same_device = same_device and w_scale.device == device
    if not same_device:
        tensors = [XQ, WQ, Y]
        if has_x_scale:
            tensors.append(x_scale)
        if has_w_scale:
            tensors.append(w_scale)
        devices = {tensor.device for tensor in tensors}
        raise ValueError(
            f"{entry}: all tensors must be on one device; got "
            f"{sorted(map(str, devices))}"
        )


@functools.lru_cache(maxsize=None)
def _require_registered_kid_cached(
    arch: str, family: str, resolved: int, output_dtype: torch.dtype
) -> int:
    """Validate one immutable exact capability and memoize successful hits.

    ``lru_cache`` does not cache exceptions, so an unavailable capability is
    re-evaluated instead of becoming a stale negative if a development test or
    future runtime extension adds registry metadata.
    """
    instance = get_kernel_instance(arch, family, resolved, output_dtype)
    if instance is None:
        raise ValueError(
            "no registered OPUS kernel for "
            f"(arch={arch!r}, family={family!r}, kid={resolved}, "
            f"Y.dtype={output_dtype})"
        )
    return resolved


def _require_registered_kid(
    *, arch: str, family: str, kid: object, output_dtype: torch.dtype
) -> int:
    try:
        resolved = int(kid)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"OPUS {family} kid must be an integer, got {kid!r}") from exc
    return _require_registered_kid_cached(arch, family, resolved, output_dtype)


# ---- Private exact-kid pybind bindings -----------------------------------


def _gen_opus_gemm_a8w8_launch_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    kid: int,
) -> Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_launch",
    gen_fake=_gen_opus_gemm_a8w8_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    kid: int,
) -> Tensor: ...


def _gen_opus_gemm_a8w8_blockscale_launch_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    kid: int,
) -> Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_blockscale_launch",
    gen_fake=_gen_opus_gemm_a8w8_blockscale_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_blockscale_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    kid: int,
) -> Tensor: ...


def _gen_opus_gemm_a8w8_blockscale_bpreshuffle_launch_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    kid: int,
) -> Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w8_blockscale_bpreshuffle_launch",
    gen_fake=_gen_opus_gemm_a8w8_blockscale_bpreshuffle_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    kid: int,
) -> Tensor: ...


# ---- Canonical public wrappers -------------------------------------------


def opus_gemm_a8w8_launch(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    kid: int = 2,
) -> Tensor:
    """Launch the gfx950 no-scale A8W8 family with an exact kid.

    ``XQ/WQ/Y`` must be contiguous 3D ``[B,M,K]``, ``[B,N,K]`` and
    ``[B,M,N]`` tensors. Inputs are FP8 and ``Y`` is FP32. The generated kid-2
    launcher additionally enforces its two-tile minimum, even K-tile loop and
    even-K prefetch contract.
    """
    entry = "opus_gemm_a8w8_launch"
    _check_same_device(entry, XQ, WQ, Y)
    arch = _device_arch(XQ.device)
    resolved = _require_registered_kid(
        arch=arch, family=_A8W8_FAMILY, kid=kid, output_dtype=Y.dtype
    )
    _opus_gemm_a8w8_launch_raw(XQ, WQ, Y, resolved)
    return Y


def opus_gemm_a8w8_blockscale_launch(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    *,
    kid: int = 1,
) -> Tensor:
    """Launch the gfx950 plain-WQ blockscale A8W8 family.

    This API accepts neither bias nor ``group_layout``. ``XQ/WQ/Y`` use the
    same 3D FP8/FP8/FP32 contract as :func:`opus_gemm_a8w8_launch`; ``WQ`` is
    plain, not pre-shuffled. Both FP32 contiguous scales are mandatory:
    ``x_scale=[B,M,K/128]`` and ``w_scale=[B,N/128,K/128]``. For ``B=1`` the
    leading scale dimension may be omitted.
    """
    entry = "opus_gemm_a8w8_blockscale_launch"
    _check_same_device(entry, XQ, WQ, Y, x_scale, w_scale)
    arch = _device_arch(XQ.device)
    resolved = _require_registered_kid(
        arch=arch,
        family=_A8W8_BLOCKSCALE_FAMILY,
        kid=kid,
        output_dtype=Y.dtype,
    )
    _opus_gemm_a8w8_blockscale_launch_raw(
        XQ, WQ, Y, x_scale, w_scale, resolved
    )
    return Y


def _lookup_opus_bpreshuffle_tuned_kid(
    XQ: Tensor,
    WQ: Tensor,
) -> int | None:
    """Read the existing A8 config and return only an OPUS winner."""
    from ...jit.core import AITER_CONFIGS
    from ..gemm_op_a8w8 import get_CKGEMM_config

    M = int(XQ.shape[-2])
    K = int(XQ.shape[-1])
    N = int(WQ.shape[-2])
    config = get_CKGEMM_config(
        M,
        N,
        K,
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE,
    )
    if config is None or str(config.get("libtype", "")).lower() != "opus":
        return None
    value = config.get("kernelId", config.get("solidx"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_bpreshuffle_kid(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    *,
    arch: str,
    kid: int | None,
) -> int:
    if kid is not None:
        candidate: object = kid
        source = "explicit"
    else:
        tuned = _lookup_opus_bpreshuffle_tuned_kid(XQ, WQ)
        if tuned is not None:
            candidate = tuned
            source = "OPUS tuned row"
        else:
            candidate = OPUS_DEFAULT_A8W8_BPRESHUFFLE_KID_BY_ARCH.get(arch)
            source = "per-arch default"

    if candidate is None:
        raise RuntimeError(
            "no registered OPUS kernel for "
            f"(arch={arch!r}, family={_A8W8_BPRESHUFFLE_FAMILY!r}, "
            f"Y.dtype={Y.dtype}); no OPUS tuned row or per-arch default exists"
        )
    try:
        return _require_registered_kid(
            arch=arch,
            family=_A8W8_BPRESHUFFLE_FAMILY,
            kid=candidate,
            output_dtype=Y.dtype,
        )
    except ValueError as exc:
        raise ValueError(f"invalid {source} for OPUS bpreshuffle: {exc}") from exc


def opus_gemm_a8w8_blockscale_bpreshuffle_launch(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
    *,
    kid: int | None = None,
) -> Tensor:
    """Launch the per-arch blockscale-bpreshuffle family.

    Resolution order is ``explicit -> OPUS A8 tuned row -> per-arch default``.
    The private raw binding always receives a resolved integer kid.

    ``WQ`` must already contain the pre-shuffled weight representation required
    by the resolved launcher. Pre-shuffle is a content/layout semantic and
    cannot be proven from Tensor shape or strides. The current gfx942 kid 11000
    requires a real ``shuffle_weight(WQ, layout=(16, 16))`` result, BF16 ``Y``,
    FP32 scales, batch 1, and exact 128-wide N/K groups. gfx950 and gfx1250
    expose this same stable API but currently have empty OPUS family tables.
    """
    entry = "opus_gemm_a8w8_blockscale_bpreshuffle_launch"
    _check_same_device(entry, XQ, WQ, Y, x_scale, w_scale)
    arch = _device_arch(XQ.device)
    if arch not in _SUPPORTED_OPUS_ARCHES:
        raise RuntimeError(
            f"{entry} requires one of {_SUPPORTED_OPUS_ARCHES}; got {arch!r}"
        )
    resolved = _resolve_bpreshuffle_kid(
        XQ, WQ, Y, arch=arch, kid=kid
    )
    _opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw(
        XQ, WQ, x_scale, w_scale, Y, resolved
    )
    return Y


# ---- Legacy Python-only compatibility ------------------------------------


def opus_gemm_a8w8_blockscale_bpreshuffle_tune(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor | None = None,
    kernelId: int = 11000,
) -> Tensor:
    """Deprecated gfx942 compatibility wrapper around the canonical launch."""
    warnings.warn(
        "opus_gemm_a8w8_blockscale_bpreshuffle_tune is deprecated; use "
        "opus_gemm_a8w8_blockscale_bpreshuffle_launch(..., kid=...) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if Y is None:
        Y = torch.empty(
            (XQ.shape[-2], WQ.shape[-2]),
            device=XQ.device,
            dtype=torch.bfloat16,
        )
    return opus_gemm_a8w8_blockscale_bpreshuffle_launch(
        XQ, WQ, x_scale, w_scale, Y, kid=kernelId
    )


__all__ = [
    "OPUS_DEFAULT_A8W8_BPRESHUFFLE_KID_BY_ARCH",
    "opus_gemm_a8w8_launch",
    "opus_gemm_a8w8_blockscale_launch",
    "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
    "opus_gemm_a8w8_blockscale_bpreshuffle_tune",
]
