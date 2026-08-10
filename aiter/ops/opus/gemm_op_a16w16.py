# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS a16w16 Python API and id-based launch wrapper.

The shape-driven entry resolves policy in Python, in the fixed order
``explicit -> tuned CSV -> per-arch heuristic -> framework fallback``.  Every
OPUS result is represented by a resolved ``LaunchConfig`` and launched through
``opus_gemm_a16w16_tune``; the generic C++ bf16 shape selector remains bound
only as a Step-1 parity probe.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from ...jit.core import compile_ops
from csrc.opus_gemm.opus_gemm_common import (
    get_kernel_instance,
    kernel_needs_external_workspace,
)

from ._selector_a16w16 import LaunchConfig, select_launch_config
from ._workspace import WorkspacePlan, allocate_workspace, validate_workspace
from ._workspace_a16w16 import plan_a16w16_workspace

_SUPPORTED_OPUS_ARCHES = ("gfx942", "gfx950", "gfx1250")


def _device_arch_and_cu(device: torch.device) -> tuple[str, int]:
    """Return the live tensor device's gfx name and CU count without caching."""
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

# ---- Low-level pybind bindings --------------------------------------------


def _gen_opus_gemm_a16w16_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None = None,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    return Y


# Raw pybind binding to the C++ id-based dispatcher. We wrap it in a Python
# function below to add a stride-layout guard before the C++ call -- the
# launcher hardcodes stride_b_batch == N*K and reads gpu memory directly,
# so a broadcast / non-contiguous WQ silently corrupts results or faults
# the GPU. Keep `gen_fake` and `fc_name` on the raw binding so dynamo and
# torch.library see the underlying op.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_tune",
    gen_fake=_gen_opus_gemm_a16w16_tune_fake_tensors,
    develop=True,
)
def _opus_gemm_a16w16_tune_raw(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias: torch.Tensor | None = None,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


def _check_a16w16_tune_layout(XQ: torch.Tensor, WQ: torch.Tensor, Y: torch.Tensor):
    """Reject layouts that the opus launcher's hardcoded strides cannot serve.

    Mirrors the kargs setup in csrc/opus_gemm/gen_instances.py
    (_gen_flatmm_splitk_instance et al.):
        kargs.stride_a        = K
        kargs.stride_b        = K
        kargs.stride_c        = N
        kargs.stride_a_batch  = M * K
        kargs.stride_b_batch  = N * K
        kargs.stride_c_batch  = M * N
    The kernel reads memory at `ptr + batch_id * stride_*_batch + ...`
    directly. Any broadcast view (batch stride == 0), transpose, or
    sliced layout will hit garbage / unmapped memory.

    Cheap to run (a handful of integer comparisons); only raised on real
    misuse so the hot path pays nothing.
    """
    for name, t in (("XQ", XQ), ("WQ", WQ), ("Y", Y)):
        if t.dim() != 3:
            raise ValueError(
                f"opus_gemm_a16w16_tune: {name} must be 3D (got "
                f"{name}.shape={tuple(t.shape)}). The C++ launcher reads "
                f"`{name}.size(0)` as batch and indexes with hardcoded "
                f"stride_*_batch == size(1)*size(2)."
            )

    batch, M, K = XQ.shape
    b_w, N, K_w = WQ.shape
    b_y, M_y, N_y = Y.shape
    if (b_w, K_w) != (batch, K):
        raise ValueError(
            f"opus_gemm_a16w16_tune: WQ shape mismatch (got "
            f"WQ.shape={tuple(WQ.shape)}, expected "
            f"({batch}, N, {K})); XQ.shape={tuple(XQ.shape)}"
        )
    if (b_y, M_y, N_y) != (batch, M, N):
        raise ValueError(
            f"opus_gemm_a16w16_tune: Y shape mismatch (got "
            f"Y.shape={tuple(Y.shape)}, expected ({batch}, {M}, {N}))"
        )

    # XQ / WQ: the K (innermost / contraction) dimension may be padded -- the
    # launcher passes the tensor's real leading stride as kargs.stride_a/stride_b
    # and the kernels use it as the lda for BOTH addressing and the gmem buffer
    # bound, so a row pitch > K (e.g. a 2880-wide tensor stored at lda 3072) is
    # served correctly. We only require:
    #   * innermost stride == 1   (the kernel layout hardcodes the K stride to 1)
    #   * row pitch (stride[1]) >= K
    #   * batch stride == rows * row pitch (or batch == 1) -- rejects broadcast
    #     (stride 0) and transposed / overlapping views.
    for name, t, rows in (("XQ", XQ, M), ("WQ", WQ, N)):
        s0, s1, s2 = t.stride()
        k_inner = t.shape[2]
        ok = s2 == 1 and s1 >= k_inner and (batch == 1 or s0 == rows * s1)
        if not ok:
            raise NotImplementedError(
                f"opus_gemm_a16w16_tune: {name} must be K-contiguous with an "
                f"optional padded leading dim -- need stride[2]==1, "
                f"stride[1]>={k_inner}, and stride[0]==size(1)*stride[1] (or "
                f"batch==1). Got {name}.stride()={tuple(t.stride())}, "
                f"{name}.shape={tuple(t.shape)}. Broadcast / transpose / "
                f"non-K-contiguous slices are not supported; materialize with "
                f"`{name} = {name}.contiguous()` before calling."
            )
    # Y is the output: the launcher hardcodes stride_c == N and
    # stride_c_batch == M*N, so it must be fully contiguous.
    y_want = (M * N, N, 1)
    if tuple(Y.stride()) != y_want:
        raise NotImplementedError(
            f"opus_gemm_a16w16_tune: Y must have contiguous strides {y_want} "
            f"(got Y.stride()={tuple(Y.stride())}, Y.shape={tuple(Y.shape)}). "
            f"The launcher hardcodes stride_c == N and stride_c_batch == M*N; "
            f"materialize with `Y = Y.contiguous()` before calling."
        )


def _prepare_a16w16_workspace(
    config: LaunchConfig,
    XQ: torch.Tensor,
    Y: torch.Tensor,
    workspace: torch.Tensor | None = None,
) -> tuple[WorkspacePlan | None, torch.Tensor | None]:
    """Plan and resolve the workspace for an already-selected launch.

    This is the single allocation/validation point for the Step-5 raw ABI.
    Caller-provided tensors go through the same validator and are returned
    unchanged; only a missing required workspace calls ``torch.empty``.
    """
    if config.is_framework_fallback or config.actual_kid is None:
        raise ValueError("cannot prepare a workspace for framework fallback")
    instance = get_kernel_instance(config.arch, config.family, config.actual_kid)
    if instance is None:
        raise RuntimeError(
            "resolved OPUS launch has no canonical instance: "
            f"({config.arch}, {config.family}, {config.actual_kid})"
        )

    batch, M, K = map(int, XQ.shape)
    N = int(Y.shape[2])
    plan = plan_a16w16_workspace(
        instance,
        arch=config.arch,
        kid=config.actual_kid,
        M=M,
        N=N,
        K=K,
        batch=batch,
        split_k=config.allocation_split_k,
    )
    if plan is None:
        if workspace is not None:
            raise ValueError(
                f"OPUS a16w16 kid {config.actual_kid} does not use an "
                "external workspace"
            )
        return None, None

    if workspace is None:
        workspace = allocate_workspace(plan, XQ.device)
    else:
        validate_workspace(workspace, plan, XQ.device)
    return plan, workspace


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
    """Prepared Step-5 ``plan -> torch.empty -> raw launch`` path.

    The current C++ binding still has the legacy no-workspace ABI, so
    production callers intentionally do not enter this helper in Step 2.
    Step 5 will pass the updated raw binding as ``raw_launch`` and switch the
    two public entry points to this centralized path.
    """
    _, workspace = _prepare_a16w16_workspace(config, XQ, Y, workspace)
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


def opus_gemm_a16w16_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    bias=None,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    """Low-level id-based dispatcher (Python guard + C++ launch).

    See module docstring. This Python wrapper checks XQ/WQ/Y layout, resolves
    gfx942 requested/actual kid and split-K semantics, and then forwards the
    concrete id to the underlying pybind binding.

    Parameters
    ----------
    bias : optional D_OUT-typed bias tensor, accepted shapes:
           [M] (broadcast across batch; requires batch==1) or [batch, M].
           Only honored on bias-aware kid ranges (split-barrier kid 4..9
           and a16w16_flatmm_splitk kid 200..299); the C++ dispatcher
           rejects bias on other kids.

    Backwards-compatibility note
    ----------------------------
    Older callers used ``opus_gemm_a16w16_tune(XQ, WQ, Y, kernelId, splitK)``
    with positional args (no bias slot). When the 4th positional argument
    is an int, we silently treat it as kernelId and shift remaining args
    accordingly so existing tuner / test scripts keep working without an
    edit. Mixed-style calls (``..., bias=t, kernelId=k``) keep their kwargs
    semantics.
    """
    # Positional-int back-compat: opus_gemm_a16w16_tune(XQ, WQ, Y, kid, splitK).
    # When `bias` arrives as an int (which torch_library would otherwise
    # reject as not Optional[Tensor]), reinterpret as kernelId.
    if isinstance(bias, int) and not isinstance(bias, bool):
        # Positional int means "this was meant to be kernelId"; treat the
        # next positional (kernelId) as splitK and the original splitK
        # (default 0) as truly unset.
        if splitK != 0 and kernelId == 0:
            # Shouldn't happen in old call sites, but be defensive.
            new_splitK = splitK
        else:
            new_splitK = kernelId
        kernelId = bias
        splitK = new_splitK
        bias = None
    _check_a16w16_tune_layout(XQ, WQ, Y)
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
        explicit_kid=int(kernelId),
        explicit_split_k=int(splitK),
    )
    if config.actual_kid is None:
        raise RuntimeError("an explicit OPUS kid cannot resolve to framework fallback")

    # C++ launcher is in-place on Y (returns void after PR #2932-style
    # refactor to aiter_tensor_t). Keep the wrapper's `return Y`
    # contract so callers that did `Y = opus_gemm_a16w16_tune(...)`
    # still see the populated Y.
    _opus_gemm_a16w16_tune_raw(
        XQ,
        WQ,
        Y,
        bias,
        config.actual_kid,
        config.launch_split_k,
    )
    return Y


# Private legacy bf16 no-scale binding retained only as a parity-golden probe
# during Step 1. Production a16w16 calls do not use it: Python resolves the kid
# first and enters the id-based tune wrapper above. It still wraps the generic
# C++ ``opus_gemm`` symbol so parity tests can compare against the old policy.
#
# Parameter annotations match the C++ signature exactly; torch_library's
# infer_schema requires every parameter be typed even though we always
# pass None for the last three.
def _gen_opus_gemm_bf16_dispatch_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    group_layout: torch.Tensor | None = None,
    x_scale: torch.Tensor | None = None,
    w_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm",
    gen_fake=_gen_opus_gemm_bf16_dispatch_fake_tensors,
    develop=True,
)
def _opus_gemm_bf16_dispatch(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    group_layout: torch.Tensor | None = None,
    x_scale: torch.Tensor | None = None,
    w_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...


# ---- High-level shape-driven API -----------------------------------------

def is_splitk_kid(kid: int) -> bool:
    """Return whether ``kid`` is a registered a16w16 workspace kernel.

    This compatibility helper has no arch argument, so it searches the three
    disjoint a16w16 registries. Capability still comes from the canonical
    instance registry rather than a copied integer band.
    """
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

    # Resolve A first so we know `batch`.
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

    # B accepted shapes:
    #   * [N, K]                       - allowed only when batch == 1
    #   * [batch, N, K] real-strided   - allowed for any batch
    #
    # The opus a16w16-family launchers hardcode `kargs.stride_b_batch = N * K`
    # (csrc/opus_gemm/gen_instances.py around lines 531/634/735/865) and the
    # device kernel computes `ptr_b + batch_id * stride_b_batch` directly,
    # ignoring the tensor's reported stride. A `B.unsqueeze(0).expand(batch,
    # -1, -1)` view has batch_stride == 0, so the kernel reads garbage past
    # B's real allocation -- this manifests as NaN, large numerical errors,
    # or HIP "Memory access fault by GPU node-1" depending on what the
    # caching allocator parked next to B. Reject the broken case at the
    # Python boundary rather than letting it through.
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
        WQ = B.unsqueeze(0)  # batch == 1 here; kernel never reads stride_b_batch.
    elif B.dim() == 3:
        b_b, N, K_b = B.shape
        if K_b != K:
            raise ValueError(f"K dimension mismatch: A has K={K}, B has K={K_b}")
        if b_b != batch:
            raise ValueError(
                f"B batch mismatch: A has batch={batch}, B has batch={b_b}"
            )
        # Reject expand-style broadcast views (batch_stride=0) up front. Any
        # other layout (contiguous, transposed N/K, etc.) is still rejected
        # below by the elements-per-row check; the launcher requires
        # B[b].stride(0) == N*K and B[b].stride(1) == K.
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

    # Bias validation. Bias may be fp32 OR match the output dtype: the gfx1250
    # splitk main kernel always writes an fp32 workspace and the reduce kernel
    # folds bias in fp32 before the final cast to Y, so an fp32 bias is exact
    # and free regardless of Y dtype (the common accuracy-friendly case for a
    # bf16 output). Bias is per-output-feature [N] (F.linear convention):
    #   * [N]          -> stride_bias_batch = 0 (broadcast across batch)
    #   * [batch, N]   -> stride_bias_batch = N
    # Matches the C++-side gfx1250 bias validation in gen_instances_gfx1250.py.
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
    """Execute the selector's terminal framework fallback into ``Y``."""
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
    """Shape-driven opus a16w16 GEMM.

    Parameters
    ----------
    A : [M, K] or [batch, M, K], bf16
    B : bf16 weight, plain layout (not pre-shuffled). Two accepted shapes:
        * [N, K]            -- requires batch == 1 (i.e. A is 2D, or A is
                               3D with leading dim 1).
        * [batch, N, K]     -- contiguous strides (N*K, K, 1) only.
                               Broadcast views (e.g. ``B.unsqueeze(0).
                               expand(batch, -1, -1)``) are rejected
                               because the opus launcher assumes
                               ``stride_b_batch == N*K``; pass
                               ``.contiguous()`` if you need to broadcast
                               a single-batch weight across A.
    bias : optional per-output-feature bias (F.linear convention), dtype
        must equal `dtype` (match_d_out). Accepted shapes:
        * [N]                  -- broadcast across batch.
        * [batch, N]           -- per-batch bias vector.
        bias is fused when the resolved OPUS kid supports it. Requests that
        the current raw tune ABI cannot serve use the terminal framework
        fallback.
    dtype : output dtype, bf16 or fp32 (any kernel family supports either)
    kernelId : optional explicit override. When given, bypass CSV/heuristic
        selection and strictly validate this instance before launch.
    splitK : optional literal KBatch; only honored when kernelId is set.
    out : optional preallocated [batch, M, N] output; reused instead of
        allocating a fresh tensor.

    Returns
    -------
    Tensor with shape [M, N] when A was 2D, [batch, M, N] when A was 3D.
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

    # Explicit, tuned, and CSV-miss heuristic choices all converge here. The
    # generic C++ shape selector is intentionally not part of production flow.
    opus_gemm_a16w16_tune(
        XQ,
        WQ,
        Y,
        bias,
        config.actual_kid,
        config.launch_split_k,
    )
    return _finalize_output(Y, reshape_out_to_2d)


# Per-stream splitk workspace init. Call once inside `with torch.cuda.stream(s):`
# (eagerly, before HIP graph capture) to register a workspace handle for that
# stream. Needed under vLLM/sglang-style TBO where two CPU threads drive two
# streams concurrently -- each captured graph must bake in its own buffer
# pointer; the prior thread_local cache would fail capture on the second
# stream. After init, run the largest expected gemm eagerly on the same
# stream to grow the buffer, then capture.
@compile_ops("module_deepgemm_opus", fc_name="opus_gemm_workspace_init", develop=True)
def opus_gemm_workspace_init() -> None: ...


__all__ = [
    "gemm_a16w16_opus",
    "is_splitk_kid",
    "opus_gemm_a16w16_tune",
    "opus_gemm_workspace_init",
]
