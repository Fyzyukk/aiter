# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS gfx950 MXFP8 A x MXFP4 B GEMM (family ``a8w4``), ``layout="bpreshuffle"``.

Operands (1x32 e8m0 scales, bf16 Y [M, N]):

* ``XQ``: fp8 [M, K] -- the row-major A, or the [M, K] view of A stored [K, M]
  (TT / TN, M % 16).
* ``WQ``: fp4x2, the ``shuffle_weight(w, layout=(16, 16))`` of B stored [N, K/2]
  (passed as [N, K/2]) or of B stored [K, N/2] (NN / TN, N % 32; passed as its
  transposed [N/2, K] view).
* ``x_scale`` / ``w_scale``: the ``e8m0_shuffle`` of [M, K/32] / [N, K/32] in
  every layout.

K % 256; any M; N % 16. Local kids (``a8w4/gfx950/<CU>``, pre-compiled, one
image per layout): 0 256x256 without M/N boundary handling, 1-6 one tile per
block, 7-11 persistent, 12-16 split-K (fp32 partials summed by
splitk_reduce_kernel). Launched through ``opus_gemm`` (``family="a8w4"``,
``layout="bpreshuffle"``); the C++ entry checks the tensor contract and the
kid's launcher picks the layout image from the XQ / WQ strides.
"""

import torch
from torch import Tensor

from csrc.opus_gemm.opus_gemm_common import get_registry

from ...jit.core import compile_ops
from ...utility import dtypes
from ._arch import _device_arch_and_cu

A8W4_FAMILY = "a8w4"
_A8W4_K_ALIGN = 256  # two K tiles of 128
_E8M0_DTYPES = (dtypes.fp8_e8m0, torch.uint8)


def _gen_opus_gemm_a8w4_bpreshuffle_launch_fake_tensors(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    workspace: Tensor | None,
    kid: int,
    split_k: int,
) -> Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a8w4_bpreshuffle_launch",
    gen_fake=_gen_opus_gemm_a8w4_bpreshuffle_launch_fake_tensors,
    develop=True,
)
def _opus_gemm_a8w4_bpreshuffle_launch_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    workspace: Tensor | None,
    kid: int,
    split_k: int,
) -> Tensor: ...


def _a8w4_workspace_bytes(instance, M: int, N: int, split_k: int) -> int:
    """fp32 partials [S][padded_M][padded_N] of a split-K launch, in bytes;
    splitk_reduce_kernel addresses them through one 32-bit buffer range."""
    padded_m = -(-M // instance.B_M) * instance.B_M
    padded_n = -(-N // instance.B_N) * instance.B_N
    return split_k * padded_m * padded_n * 4


def a8w4_kid_runs(
    instance, M: int, N: int, K: int, split_k: int = 0, layout: str = "NT"
) -> bool:
    """Whether a8w4 kid ``instance`` runs (M, N, K) in ``layout`` (NT / NN / TT /
    TN) with ``split_k``: the kid's alignment, the 32-bit buffer ranges of a
    transposed operand and of one C tile, and a split-K workspace under 4 GiB."""
    layout = layout.upper()
    a_t, b_t = layout[0] == "T", layout[1] == "N"
    if (a_t and M % 16) or (b_t and N % 32) or K <= 0 or K % _A8W4_K_ALIGN:
        return False
    if (
        max(M * K if a_t else 0, N // 2 * K if b_t else 0, 2 * instance.B_M * N)
        > (1 << 31) - 1
    ):
        return False
    if instance.co_variant == "nob":
        if M % instance.B_M or N % instance.B_N:
            return False
    elif N % 16:
        return False
    if instance.co_variant == "sk":
        return split_k <= 1 or (
            K % (split_k * _A8W4_K_ALIGN) == 0
            and _a8w4_workspace_bytes(instance, M, N, split_k) <= (1 << 32) - 1
        )
    return split_k <= 1


def _a8w4_plan(instance, M: int, N: int, K: int, split_k: int, cu_num: int):
    """(ABI split_k, fp32 workspace elements; 0 = none) of one launch. Split-K
    kids with split_k = 0 take the largest S in {8, 4, 2} with tiles * S <= CUs,
    K % (S * 256) == 0 and a workspace under 4 GiB; the launcher checks the rest."""
    if instance.co_variant != "sk":
        return split_k, 0
    padded_m = -(-M // instance.B_M) * instance.B_M
    padded_n = -(-N // instance.B_N) * instance.B_N
    tiles = (padded_m // instance.B_M) * (padded_n // instance.B_N)
    S = split_k or next(
        (
            s
            for s in (8, 4, 2)
            if tiles * s <= cu_num
            and K % (s * _A8W4_K_ALIGN) == 0
            and _a8w4_workspace_bytes(instance, M, N, s) <= (1 << 32) - 1
        ),
        1,
    )
    return S, S * padded_m * padded_n if S > 1 else 0


def _launch_a8w4_gemm(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    *,
    kid: int,
    split_k: int = 0,
    workspace: Tensor | None = None,
    route_arch: str | None = None,
) -> Tensor:
    """Launch logical 2D MXFP8 A x MXFP4 B GEMM (gfx950); ``split_k``: split-K
    kids only, 0 = auto (see :func:`_a8w4_plan`)."""
    if any(t.dim() != 2 for t in (XQ, WQ, Y, x_scale, w_scale)):
        raise ValueError(
            "opus_gemm a8w4 expects logical 2D XQ/WQ/Y/x_scale/w_scale; "
            "this family is GEMM-only"
        )
    if x_scale.dtype not in _E8M0_DTYPES or w_scale.dtype not in _E8M0_DTYPES:
        raise ValueError(
            f"OPUS a8w4 requires e8m0 scales; got {x_scale.dtype}/{w_scale.dtype}"
        )
    device_arch, cu_num = _device_arch_and_cu(XQ.device)
    instance = get_registry(A8W4_FAMILY, route_arch or device_arch, cu_num)[int(kid)]
    m, k = XQ.shape
    n = 2 * WQ.shape[0] if WQ.shape[1] == k else WQ.shape[0]
    abi_split_k, ws_numel = _a8w4_plan(instance, m, n, k, int(split_k), cu_num)
    if ws_numel and workspace is None:
        workspace = torch.empty(ws_numel, dtype=torch.float32, device=XQ.device)
    with torch.cuda.device(XQ.device):
        _opus_gemm_a8w4_bpreshuffle_launch_raw(
            XQ.unsqueeze(0),
            WQ.unsqueeze(0),
            Y.unsqueeze(0),
            x_scale,
            w_scale,
            workspace,
            int(kid),
            abi_split_k,
        )
    return Y


def _validate_a8w4_public_contract(
    *,
    kid: int,
    input_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    output_dtype: torch.dtype,
    layout: str,
    has_x_scale: bool,
    has_w_scale: bool,
    has_bias: bool,
) -> None:
    """Immutable per-kid contract checked (and cached) by ``dispatch._resolve_contract``."""
    if input_dtype != dtypes.fp8 or weight_dtype != dtypes.fp4x2:
        raise ValueError(
            f"OPUS a8w4 kid {kid} requires fp8 XQ and fp4x2 WQ; got "
            f"{input_dtype}/{weight_dtype}"
        )
    if output_dtype != torch.bfloat16:
        raise ValueError(f"OPUS a8w4 kid {kid} requires bf16 Y; got {output_dtype}")
    if layout != "bpreshuffle":
        raise ValueError(
            f"OPUS kid {kid} belongs to family a8w4 and requires "
            f"layout='bpreshuffle'; got {layout!r}"
        )
    if not (has_x_scale and has_w_scale):
        raise ValueError("OPUS a8w4 requires x_scale and w_scale")
    if has_bias:
        raise ValueError("OPUS family a8w4 does not support bias")


__all__ = ["a8w4_kid_runs"]
