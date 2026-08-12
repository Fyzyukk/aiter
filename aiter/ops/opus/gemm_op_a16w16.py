# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS a16w16 selection and Torch workspace launch APIs."""

import warnings
from collections.abc import Callable

import torch
from torch import Tensor

from ...jit.core import compile_ops
from csrc.opus_gemm.opus_gemm_common import (
    get_kernel_instance,
    kernel_needs_external_workspace,
)

from ._selector_a16w16 import LaunchConfig, select_launch_config

_SUPPORTED_OPUS_ARCHES = ("gfx942", "gfx950", "gfx1250")
_WORKSPACE_DTYPES = {
    "bf16_t": torch.bfloat16,
    "fp32_t": torch.float32,
}


def _device_arch_and_cu(device: torch.device) -> tuple[str, int]:
    """Return the device architecture and CU count."""
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
    return arch, int(props.multi_processor_count)


# ---- Low-level bindings ---------------------------------------------------


def _gen_opus_gemm_a16w16_launch_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_launch",
    gen_fake=_gen_opus_gemm_a16w16_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a16w16_launch_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> torch.Tensor: ...


# Keep pybind for compatibility/A-B; production uses C ABI.


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_launch_cabi",
    ffi_type="ctypes",
    # Reuse the mixed pybind module's shared object.
    ctypes_force_torch_exclude=False,
)
def _opus_gemm_a16w16_launch_ctypes_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    workspace: torch.Tensor | None,
    kid: int,
    split_k: int,
) -> None: ...


def _check_a16w16_launch_layout(XQ: torch.Tensor, WQ: torch.Tensor, Y: torch.Tensor):
    """Validate launcher-required 3D shapes and strides."""
    for name, t in (("XQ", XQ), ("WQ", WQ), ("Y", Y)):
        if t.dim() != 3:
            raise ValueError(
                f"opus_gemm_a16w16_launch: {name} must be 3D (got "
                f"{name}.shape={tuple(t.shape)}). The C++ launcher reads "
                f"`{name}.size(0)` as batch and indexes with hardcoded "
                f"stride_*_batch == size(1)*size(2)."
            )

    batch, M, K = XQ.shape
    b_w, N, K_w = WQ.shape
    b_y, M_y, N_y = Y.shape
    if (b_w, K_w) != (batch, K):
        raise ValueError(
            f"opus_gemm_a16w16_launch: WQ shape mismatch (got "
            f"WQ.shape={tuple(WQ.shape)}, expected "
            f"({batch}, N, {K})); XQ.shape={tuple(XQ.shape)}"
        )
    if (b_y, M_y, N_y) != (batch, M, N):
        raise ValueError(
            f"opus_gemm_a16w16_launch: Y shape mismatch (got "
            f"Y.shape={tuple(Y.shape)}, expected ({batch}, {M}, {N}))"
        )

    # XQ/WQ allow padded rows but require contiguous K and dense batches.
    for name, t, rows in (("XQ", XQ, M), ("WQ", WQ, N)):
        s0, s1, s2 = t.stride()
        k_inner = t.shape[2]
        ok = s2 == 1 and s1 >= k_inner and (batch == 1 or s0 == rows * s1)
        if not ok:
            raise NotImplementedError(
                f"opus_gemm_a16w16_launch: {name} must be K-contiguous with an "
                f"optional padded leading dim -- need stride[2]==1, "
                f"stride[1]>={k_inner}, and stride[0]==size(1)*stride[1] (or "
                f"batch==1). Got {name}.stride()={tuple(t.stride())}, "
                f"{name}.shape={tuple(t.shape)}. Broadcast / transpose / "
                f"non-K-contiguous slices are not supported; materialize with "
                f"`{name} = {name}.contiguous()` before calling."
            )
    # Y must match the launcher's contiguous output strides.
    y_want = (M * N, N, 1)
    if tuple(Y.stride()) != y_want:
        raise NotImplementedError(
            f"opus_gemm_a16w16_launch: Y must have contiguous strides {y_want} "
            f"(got Y.stride()={tuple(Y.stride())}, Y.shape={tuple(Y.shape)}). "
            f"The launcher hardcodes stride_c == N and stride_c_batch == M*N; "
            f"materialize with `Y = Y.contiguous()` before calling."
        )


def _init_a16w16_workspace(
    config: LaunchConfig,
    XQ: torch.Tensor,
    Y: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Prepare workspace for ``config.actual_kid``."""
    if config.is_framework_fallback or config.actual_kid is None:
        raise ValueError(
            "opus_gemm_a16w16_launch: cannot initialize a workspace for "
            "framework fallback"
        )
    instance = get_kernel_instance(config.arch, config.family, config.actual_kid)
    if instance is None:
        raise RuntimeError(
            "opus_gemm_a16w16_launch: resolved launch has no canonical "
            "instance: "
            f"({config.arch}, {config.family}, {config.actual_kid})"
        )

    if not kernel_needs_external_workspace(
        config.arch, config.family, config.actual_kid
    ):
        if workspace is not None:
            raise ValueError(
                f"opus_gemm_a16w16_launch: kid {config.actual_kid} does not use an "
                "external workspace"
            )
        return None

    batch, M, K = map(int, XQ.shape)
    N = int(Y.shape[2])
    is_fused = (
        instance.kernel_tag == "a16w16_clusterlaunch_tdm_splitk_fuse"
    )
    # Fused split-K comes from the kernel entry; other kernels use selector output.
    split_k = (
        int(instance.fuse_split_k)
        if is_fused
        else int(config.allocation_split_k)
    )
    if split_k <= 0:
        raise ValueError(
            "opus_gemm_a16w16_launch: allocation split_k must be positive, "
            f"got {split_k}"
        )

    block_m = int(instance.B_M)
    block_n = int(instance.B_N)
    block_k = int(instance.B_K)
    max_useful_split_k = (K + block_k - 1) // block_k
    if split_k > max_useful_split_k:
        raise ValueError(
            "opus_gemm_a16w16_launch: "
            f"allocation split_k={split_k} exceeds the per-kid K-tile limit "
            f"{max_useful_split_k} for K={K}, B_K={block_k}"
        )

    if config.arch == "gfx1250":
        if batch != 1:
            raise ValueError(
                "opus_gemm_a16w16_launch: gfx1250 workspace kids require "
                "batch=1; "
                f"got batch={batch}"
            )
        num_tiles_m = (M + block_m - 1) // block_m
        num_tiles_n = (N + block_n - 1) // block_n
        if is_fused:
            if split_k < 2:
                raise ValueError(
                    "opus_gemm_a16w16_launch: "
                    f"gfx1250 fused kid {config.actual_kid} must declare "
                    f"compile-time SplitK >= 2, got {split_k}"
                )
            # Fused layout: M tile, N tile, published partial,
            # M element, N element.
            shape = (
                num_tiles_m,
                num_tiles_n,
                split_k - 1,
                block_m,
                block_n,
            )
        else:
            padded_m = num_tiles_m * block_m
            padded_n = num_tiles_n * block_n
            shape = (split_k, padded_m, padded_n)
    else:
        padded_m = ((M + block_m - 1) // block_m) * block_m
        padded_n = ((N + block_n - 1) // block_n) * block_n
        shape = (split_k, batch, padded_m, padded_n)

    dtype_token = instance.splitk_workspace_dtype
    try:
        dtype = _WORKSPACE_DTYPES[dtype_token]
    except KeyError as exc:
        raise ValueError(
            "opus_gemm_a16w16_launch: "
            f"workspace kid {config.actual_kid} must declare "
            f"bf16_t or fp32_t storage, got {dtype_token!r}"
        ) from exc

    required_numel = 1
    max_numel = (2**63 - 1) // int(dtype.itemsize)
    for extent in shape:
        if extent <= 0 or required_numel > max_numel // extent:
            raise OverflowError(
                "opus_gemm_a16w16_launch: "
                f"workspace shape {shape} exceeds the supported tensor size "
                f"for dtype {dtype}"
            )
        required_numel *= extent

    if workspace is not None:
        return workspace
    return torch.empty(shape, dtype=dtype, device=XQ.device)


def _launch_a16w16_with_torch_workspace(
    raw_launch: Callable[..., object],
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    config: LaunchConfig,
    *,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Prepare workspace and launch the resolved kid."""
    _check_a16w16_launch_layout(XQ, WQ, Y)
    workspace = _init_a16w16_workspace(config, XQ, Y, workspace)
    raw_launch(
        XQ,
        WQ,
        Y,
        bias,
        workspace,
        config.actual_kid,
        config.launch_split_k,
    )
    return Y


def _explicit_a16w16_launch(
    raw_launch: Callable[..., object],
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None,
    kid: int,
    split_k: int,
    *,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve and launch an explicit kid."""
    _check_a16w16_launch_layout(XQ, WQ, Y)
    batch, M, K = XQ.shape
    N = Y.shape[2]
    arch, cu_num = _device_arch_and_cu(XQ.device)
    config = select_launch_config(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=batch,
        cu_num=cu_num,
        has_bias=bias is not None,
        input_dtype=XQ.dtype,
        output_dtype=Y.dtype,
        explicit_kid=int(kid),
        explicit_split_k=int(split_k),
    )
    if config.actual_kid is None:
        raise RuntimeError(
            "opus_gemm_a16w16_launch: an explicit kid cannot resolve to "
            "framework fallback"
        )

    return _launch_a16w16_with_torch_workspace(
        raw_launch,
        XQ,
        WQ,
        Y,
        bias,
        config,
        workspace=workspace,
    )


def opus_gemm_a16w16_launch(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    kid: int,
    split_k: int = 0,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Resolve and launch an explicit a16w16 kid."""
    return _explicit_a16w16_launch(
        _opus_gemm_a16w16_launch_ctypes_raw,
        XQ,
        WQ,
        Y,
        bias,
        kid,
        split_k,
        workspace=workspace,
    )


def opus_gemm_a16w16_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias=None,
    kernelId: int = 0,
    splitK: int = 0,
    *,
    workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward deprecated calls to ``opus_gemm_a16w16_launch``."""
    # Support legacy (XQ, WQ, Y, kernelId, splitK) calls.
    if isinstance(bias, int) and not isinstance(bias, bool):
        if splitK != 0 and kernelId == 0:
            new_splitK = splitK
        else:
            new_splitK = kernelId
        kernelId = bias
        splitK = new_splitK
        bias = None
    warnings.warn(
        "opus_gemm_a16w16_tune is deprecated; use "
        "opus_gemm_a16w16_launch(..., kid=..., split_k=...) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return opus_gemm_a16w16_launch(
        XQ,
        WQ,
        Y,
        bias,
        kid=kernelId,
        split_k=splitK,
        workspace=workspace,
    )


# ---- High-level shape-driven API -----------------------------------------

def is_splitk_kid(kid: int) -> bool:
    """Return whether ``kid`` is a registered a16w16 workspace kernel."""
    for arch in _SUPPORTED_OPUS_ARCHES:
        if get_kernel_instance(arch, "a16w16", kid) is not None:
            return kernel_needs_external_workspace(arch, "a16w16", kid)
    return False


# Back-compat: the old gfx950-only single-band constants some callers imported.
_SPLITK_KID_MIN = 200
_SPLITK_KID_MAX = 299


def _validate_and_reshape(A: Tensor, B: Tensor, bias, dtype, out):
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise NotImplementedError(
            f"gemm_a16w16_opus only supports bf16 A/B "
            f"(got A.dtype={A.dtype}, B.dtype={B.dtype})."
        )
    if dtype not in (torch.bfloat16, torch.float32):
        raise NotImplementedError(
            f"gemm_a16w16_opus only supports bf16/fp32 output dtype, got {dtype}"
        )

    # Normalize A and determine batch.
    if A.dim() == 2:
        M, K = A.shape
        batch = 1
        XQ = A.unsqueeze(0)
        reshape_out_to_2d = True
    elif A.dim() == 3:
        batch, M, K = A.shape
        XQ = A
        reshape_out_to_2d = False
    else:
        raise ValueError(f"A must be 2D or 3D, got shape {tuple(A.shape)}")

    # B is [N, K] for batch 1 or dense [batch, N, K]; broadcasts are unsafe.
    if B.dim() == 2:
        N, K_b = B.shape
        if K_b != K:
            raise ValueError(f"K dimension mismatch: A has K={K}, B has K={K_b}")
        if batch > 1:
            raise NotImplementedError(
                f"gemm_a16w16_opus: B must be 3D [batch, N, K] when A is "
                f"batched (got A.shape={tuple(A.shape)}, "
                f"B.shape={tuple(B.shape)}). The opus a16w16 launchers "
                f"assume stride_b_batch == N*K (see "
                f"csrc/opus_gemm/gen_instances.py), which is incompatible "
                f"with the batch_stride=0 view a B.unsqueeze(0)."
                f"expand(batch, -1, -1) would produce. Two valid fixes:\n"
                f"  1. Broadcast explicitly:  B = B.expand({batch}, -1, "
                f"-1).contiguous()\n"
                f"  2. Pass a real 3D weight: B with shape ({batch}, N, K)"
            )
        WQ = B.unsqueeze(0)  # Batch stride is unused when batch == 1.
    elif B.dim() == 3:
        b_b, N, K_b = B.shape
        if K_b != K:
            raise ValueError(f"K dimension mismatch: A has K={K}, B has K={K_b}")
        if b_b != batch:
            raise ValueError(
                f"B batch mismatch: A has batch={batch}, B has batch={b_b}"
            )
        # Reject broadcasts and other non-dense batch layouts.
        bs0, bs1, bs2 = B.stride()
        if bs0 != N * K or bs1 != K or bs2 != 1:
            raise NotImplementedError(
                f"gemm_a16w16_opus: B must be a contiguous 3D tensor with "
                f"strides (N*K, K, 1) (got B.shape={tuple(B.shape)}, "
                f"B.stride()={tuple(B.stride())}). The opus launchers "
                f"hardcode stride_b_batch == N*K and stride_b == K; any "
                f"non-standard layout (broadcast view, transpose, slice) "
                f"will produce wrong results or a memory access fault. "
                f"Materialize via B = B.contiguous() first."
            )
        WQ = B
    else:
        raise ValueError(
            f"B must be 2D [N, K] or 3D [batch, N, K] (got shape " f"{tuple(B.shape)})"
        )

    if out is not None:
        Y = out
    else:
        Y = torch.empty(batch, M, N, dtype=dtype, device=A.device)

    # Bias must be contiguous fp32/output dtype with shape [N] or [batch, N].
    if bias is not None:
        if bias.dtype not in (dtype, torch.float32):
            raise ValueError(
                f"gemm_a16w16_opus: bias dtype must be fp32 or match output "
                f"dtype (got bias.dtype={bias.dtype}, dtype={dtype})"
            )
        if not bias.is_contiguous():
            raise ValueError(
                f"gemm_a16w16_opus: bias must be contiguous (got "
                f"bias.stride()={tuple(bias.stride())})"
            )
        if bias.dim() == 1:
            if bias.shape[0] != N:
                raise ValueError(
                    f"gemm_a16w16_opus: 1D bias length must equal N (got "
                    f"bias.shape={tuple(bias.shape)}, N={N})"
                )
        elif bias.dim() == 2:
            if tuple(bias.shape) != (batch, N):
                raise ValueError(
                    f"gemm_a16w16_opus: 2D bias must be [batch, N] (got "
                    f"bias.shape={tuple(bias.shape)}, batch={batch}, N={N})"
                )
        else:
            raise ValueError(
                f"gemm_a16w16_opus: bias must be 1D [N] or 2D [batch, N] "
                f"(got bias.shape={tuple(bias.shape)})"
            )

    return XQ, WQ, Y, M, N, K, batch, reshape_out_to_2d


def _finalize_output(Y: Tensor, reshape_out_to_2d: bool) -> Tensor:
    return Y.squeeze(0) if reshape_out_to_2d else Y


def _framework_a16w16(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    bias: Tensor | None,
) -> None:
    """Run the Torch fallback into ``Y``."""
    if Y.dtype == torch.float32:
        result = torch.bmm(XQ.float(), WQ.float().transpose(1, 2))
    else:
        result = torch.bmm(XQ, WQ.transpose(1, 2))
    if bias is not None:
        if bias.dim() == 1:
            result = result + bias.view(1, 1, -1)
        else:
            result = result + bias.unsqueeze(1)
    Y.copy_(result.to(Y.dtype))


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
    """Run shape-selected bf16 a16w16 GEMM.

    A is ``[M,K]`` or ``[batch,M,K]``. B is ``[N,K]`` for batch 1 or dense
    ``[batch,N,K]``. Bias is contiguous fp32/output-dtype ``[N]`` or
    ``[batch,N]``. Output is bf16/fp32 and follows A's rank. ``kernelId``
    selects a strict explicit launch; ``splitK`` configures it. ``out`` reuses
    a 3D output tensor.
    """
    XQ, WQ, Y, M, N, K, batch, reshape_out_to_2d = _validate_and_reshape(
        A, B, bias, dtype, out
    )
    arch, cu_num = _device_arch_and_cu(XQ.device)
    config = select_launch_config(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=batch,
        cu_num=cu_num,
        has_bias=bias is not None,
        input_dtype=A.dtype,
        output_dtype=dtype,
        explicit_kid=kernelId,
        explicit_split_k=splitK,
    )
    if config.is_framework_fallback:
        _framework_a16w16(XQ, WQ, Y, bias)
        return _finalize_output(Y, reshape_out_to_2d)

    # Launch the resolved actual kid without selecting again.
    _launch_a16w16_with_torch_workspace(
        _opus_gemm_a16w16_launch_ctypes_raw,
        XQ,
        WQ,
        Y,
        bias,
        config,
    )
    return _finalize_output(Y, reshape_out_to_2d)


def opus_gemm_workspace_init() -> None:
    """Deprecated no-op; workspace setup is automatic."""
    warnings.warn(
        "opus_gemm_workspace_init() is deprecated and no longer required; "
        "OPUS split-K workspaces are allocated by the Python wrapper",
        DeprecationWarning,
        stacklevel=2,
    )


__all__ = [
    "gemm_a16w16_opus",
    "is_splitk_kid",
    "opus_gemm_a16w16_launch",
    "opus_gemm_a16w16_tune",
    "opus_gemm_workspace_init",
]
