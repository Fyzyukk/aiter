# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-side tests for OPUS typed workspace planning and validation."""

from __future__ import annotations

import importlib
import sys

import pytest
import torch

from csrc.opus_gemm.opus_gemm_common import get_kernel_instance

from aiter.ops.opus._selector_a16w16 import LaunchConfig, select_launch_config
from aiter.ops.opus._workspace import (
    WorkspacePlan,
    allocate_workspace,
    checked_numel,
    validate_workspace,
)
from aiter.ops.opus._workspace_a16w16 import plan_a16w16_workspace


def _instance(arch: str, kid: int):
    instance = get_kernel_instance(arch, "a16w16", kid)
    assert instance is not None
    return instance


def _plan(
    arch: str,
    kid: int,
    *,
    M: int,
    N: int,
    K: int,
    batch: int,
    split_k: int,
) -> WorkspacePlan | None:
    return plan_a16w16_workspace(
        _instance(arch, kid),
        arch=arch,
        kid=kid,
        M=M,
        N=N,
        K=K,
        batch=batch,
        split_k=split_k,
    )


def _workspace_config(
    *,
    arch: str = "gfx950",
    kid: int = 200,
    allocation_split_k: int = 2,
    launch_split_k: int = 2,
) -> LaunchConfig:
    return LaunchConfig(
        arch=arch,
        family="a16w16",
        source="explicit",
        requested_kid=kid,
        actual_kid=kid,
        requested_split_k=allocation_split_k,
        allocation_split_k=allocation_split_k,
        launch_split_k=launch_split_k,
    )


def test_workspace_plan_validates_its_dynamic_contract():
    plan = WorkspacePlan(
        shape=(2, 3, 4),
        dtype=torch.float32,
        required_numel=23,
        alignment=16,
    )
    assert plan.shape == (2, 3, 4)
    assert plan.allocation_numel == 24

    with pytest.raises(ValueError, match="smaller than required_numel"):
        WorkspacePlan((2, 3), torch.float32, 7, 16)
    with pytest.raises(ValueError, match="power of two"):
        WorkspacePlan((2, 3), torch.float32, 6, 12)
    with pytest.raises(ValueError, match="must be positive"):
        WorkspacePlan((2, 0), torch.float32, 1, 16)


def test_checked_numel_rejects_extent_overflow():
    assert checked_numel((2, 3, 5), limit=30) == 30
    with pytest.raises(OverflowError, match="supported limit"):
        checked_numel((sys.maxsize, 2))


def test_allocate_workspace_is_typed_shaped_and_not_cached():
    plan = WorkspacePlan((2, 3, 4), torch.bfloat16, 24, 16)
    first = allocate_workspace(plan, torch.device("cpu"))
    second = allocate_workspace(plan, torch.device("cpu"))

    assert first.shape == plan.shape
    assert first.dtype == torch.bfloat16
    assert first.device.type == "cpu"
    assert first is not second
    assert first.data_ptr() != second.data_ptr()


def test_validate_workspace_accepts_exact_or_larger_flat_capacity():
    plan = WorkspacePlan((2, 4), torch.float32, 8, 16)
    exact = torch.empty(8, dtype=torch.float32)
    larger = torch.empty(11, dtype=torch.float32)

    assert validate_workspace(exact, plan, exact.device) is exact
    assert validate_workspace(larger, plan, larger.device) is larger


def test_validate_workspace_rejects_one_element_short():
    plan = WorkspacePlan((8,), torch.float32, 8, 16)
    with pytest.raises(ValueError, match="need at least 8 elements, got 7"):
        validate_workspace(torch.empty(7, dtype=torch.float32), plan, "cpu")


def test_validate_workspace_rejects_wrong_dtype():
    plan = WorkspacePlan((8,), torch.float32, 8, 16)
    with pytest.raises(ValueError, match="dtype mismatch"):
        validate_workspace(torch.empty(8, dtype=torch.bfloat16), plan, "cpu")


def test_validate_workspace_rejects_wrong_device():
    plan = WorkspacePlan((8,), torch.float32, 8, 16)
    workspace = torch.empty(8, dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="device mismatch"):
        validate_workspace(workspace, plan, "cpu")


def test_validate_workspace_rejects_noncontiguous_tensor():
    plan = WorkspacePlan((8,), torch.float32, 8, 16)
    workspace = torch.empty((2, 4), dtype=torch.float32).transpose(0, 1)
    with pytest.raises(ValueError, match="must be contiguous"):
        validate_workspace(workspace, plan, "cpu")


def test_validate_workspace_rejects_misaligned_storage_offset():
    plan = WorkspacePlan((8,), torch.float32, 8, 16)
    workspace = torch.empty(9, dtype=torch.float32)[1:]
    assert workspace.is_contiguous()
    assert workspace.data_ptr() % plan.alignment != 0
    with pytest.raises(ValueError, match="not sufficiently aligned"):
        validate_workspace(workspace, plan, "cpu")


@pytest.mark.parametrize(
    ("arch", "kid", "M", "N", "K", "batch", "split_k", "shape", "dtype"),
    [
        (
            "gfx950",
            200,
            65,
            33,
            4096,
            2,
            3,
            (3, 2, 128, 64),
            torch.float32,
        ),
        (
            "gfx942",
            10200,
            129,
            129,
            4096,
            2,
            3,
            (3, 2, 256, 256),
            torch.float32,
        ),
        (
            "gfx942",
            10210,
            129,
            384,
            4096,
            2,
            3,
            (3, 2, 256, 384),
            torch.bfloat16,
        ),
        (
            "gfx1250",
            20000,
            17,
            33,
            4096,
            1,
            3,
            (3, 32, 64),
            torch.float32,
        ),
    ],
)
def test_a16w16_workspace_plan_uses_instance_tile_and_dtype(
    arch, kid, M, N, K, batch, split_k, shape, dtype
):
    plan = _plan(
        arch,
        kid,
        M=M,
        N=N,
        K=K,
        batch=batch,
        split_k=split_k,
    )
    assert plan is not None
    assert plan.shape == shape
    assert plan.dtype == dtype
    assert plan.required_numel == plan.allocation_numel
    assert plan.alignment == 16


@pytest.mark.parametrize(
    ("arch", "kid"),
    [
        ("gfx950", 300),
        ("gfx942", 10000),
        # Despite being an atomic split path, this kid has no external partial
        # workspace and must not be classified by its name or numeric band.
        ("gfx942", 10310),
    ],
)
def test_non_workspace_a16w16_instances_return_none(arch, kid):
    assert (
        _plan(arch, kid, M=64, N=64, K=4096, batch=1, split_k=1) is None
    )


def test_gfx942_redirect_plan_reads_actual_not_requested_instance():
    config = select_launch_config(
        arch="gfx942",
        M=128,
        N=768,
        K=4096,
        batch=1,
        cu_num=304,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        explicit_kid=10210,
        explicit_split_k=3,
        tuned_lookup=lambda **_kwargs: None,
    )
    assert (config.requested_kid, config.actual_kid) == (10210, 10200)

    plan = plan_a16w16_workspace(
        _instance("gfx942", config.actual_kid),
        arch=config.arch,
        kid=config.actual_kid,
        M=128,
        N=768,
        K=4096,
        batch=1,
        split_k=config.allocation_split_k,
    )
    assert plan is not None
    assert plan.dtype == torch.float32

    with pytest.raises(ValueError, match="canonical actual"):
        plan_a16w16_workspace(
            _instance("gfx942", config.requested_kid),
            arch=config.arch,
            kid=config.actual_kid,
            M=128,
            N=768,
            K=4096,
            batch=1,
            split_k=config.allocation_split_k,
        )


def test_gfx1250_workspace_plan_rejects_batch_greater_than_one():
    with pytest.raises(ValueError, match="require batch=1"):
        _plan("gfx1250", 20000, M=32, N=64, K=4096, batch=2, split_k=3)


def test_workspace_plan_rejects_split_k_above_per_kid_k_tile_limit():
    with pytest.raises(ValueError, match="exceeds the per-kid K-tile limit 2"):
        _plan("gfx950", 200, M=64, N=64, K=128, batch=1, split_k=3)


def test_explicit_workspace_reuses_shared_validation_without_allocating(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config()
    XQ = torch.empty((1, 65, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 65, 33), dtype=torch.bfloat16)
    plan = _plan("gfx950", 200, M=65, N=33, K=128, batch=1, split_k=2)
    assert plan is not None
    workspace = torch.empty(plan.required_numel, dtype=plan.dtype)

    def must_not_allocate(*_args, **_kwargs):
        raise AssertionError("explicit workspace triggered allocation")

    monkeypatch.setattr(gemm, "allocate_workspace", must_not_allocate)
    resolved_plan, resolved_workspace = gemm._prepare_a16w16_workspace(
        config, XQ, Y, workspace
    )
    assert resolved_plan == plan
    assert resolved_workspace is workspace


def test_explicit_workspace_uses_shared_dtype_validation():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config()
    XQ = torch.empty((1, 65, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 65, 33), dtype=torch.bfloat16)
    wrong = torch.empty(16384, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="dtype mismatch"):
        gemm._prepare_a16w16_workspace(config, XQ, Y, wrong)


def test_prepared_step5_launch_path_allocates_and_passes_workspace():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config()
    XQ = torch.empty((1, 65, 128), dtype=torch.bfloat16)
    WQ = torch.empty((1, 33, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 65, 33), dtype=torch.bfloat16)
    calls = []

    def fake_raw(XQ_, WQ_, Y_, bias_, workspace_, kid_, split_k_):
        calls.append((XQ_, WQ_, Y_, bias_, workspace_, kid_, split_k_))

    result = gemm._launch_a16w16_with_torch_workspace(
        fake_raw, XQ, WQ, Y, None, config
    )
    assert result is Y
    assert len(calls) == 1
    assert calls[0][:4] == (XQ, WQ, Y, None)
    assert calls[0][4].shape == (2, 1, 128, 64)
    assert calls[0][4].dtype == torch.float32
    assert calls[0][5:] == (200, 2)


def test_production_path_allocates_and_passes_call_scoped_workspace(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    common = importlib.import_module("aiter.ops.opus.common")
    monkeypatch.setattr(gemm, "_device_arch_and_cu", lambda _device: ("gfx950", 256))
    monkeypatch.setattr(common, "lookup_tuned", lambda **_kwargs: None)

    calls = []

    def fake_raw(XQ, WQ, Y, bias, workspace, kernelId, splitK):
        calls.append((XQ, WQ, Y, bias, workspace, kernelId, splitK))
        Y.zero_()
        return Y

    monkeypatch.setattr(gemm, "_opus_gemm_a16w16_tune_raw", fake_raw)

    output = gemm.gemm_a16w16_opus(
        torch.empty((65, 512), dtype=torch.bfloat16),
        torch.empty((33, 512), dtype=torch.bfloat16),
    )
    assert output.shape == (65, 33)
    assert len(calls) == 1
    XQ, WQ, Y, bias, workspace, kid, split_k = calls[0]
    assert (tuple(XQ.shape), tuple(WQ.shape), tuple(Y.shape)) == (
        (1, 65, 512),
        (1, 33, 512),
        (1, 65, 33),
    )
    assert bias is None
    assert kid == 200
    assert split_k == 0
    assert workspace.shape == (1, 1, 128, 64)
    assert workspace.dtype == torch.float32
    assert workspace.device.type == "cpu"


def test_two_automatic_launches_do_not_share_a_workspace_tensor():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config()
    XQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    WQ = torch.empty((1, 64, 512), dtype=torch.bfloat16)
    workspaces = []

    def fake_raw(_XQ, _WQ, _Y, _bias, workspace, _kid, _split_k):
        workspaces.append(workspace)

    for _ in range(2):
        Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
        gemm._launch_a16w16_with_torch_workspace(
            fake_raw, XQ, WQ, Y, None, config
        )

    assert len(workspaces) == 2
    assert workspaces[0] is not workspaces[1]
    assert workspaces[0].data_ptr() != workspaces[1].data_ptr()


def test_split_k_limit_fails_before_torch_empty(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config(allocation_split_k=3, launch_split_k=2)
    XQ = torch.empty((1, 64, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 64, 64), dtype=torch.bfloat16)
    allocations = 0

    def must_not_allocate(*_args, **_kwargs):
        nonlocal allocations
        allocations += 1
        raise AssertionError("invalid split-K reached torch.empty")

    monkeypatch.setattr(gemm, "allocate_workspace", must_not_allocate)
    with pytest.raises(ValueError, match="exceeds the per-kid K-tile limit 2"):
        gemm._prepare_a16w16_workspace(config, XQ, Y)
    assert allocations == 0


def test_non_workspace_kid_rejects_explicit_workspace():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config(kid=300, allocation_split_k=1, launch_split_k=0)
    XQ = torch.empty((1, 256, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 256, 256), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="does not use an external workspace"):
        gemm._prepare_a16w16_workspace(
            config, XQ, Y, torch.empty(1, dtype=torch.float32)
        )


def _runtime_arch(device: int | None = None) -> str | None:
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()


_RAW_CASES = {
    "gfx950": dict(kid=200, M=64, N=64, K=512, split_k=2),
    "gfx942": dict(kid=10200, M=128, N=128, K=512, split_k=2),
    "gfx1250": dict(kid=20000, M=16, N=32, K=512, split_k=2),
}


def _make_raw_case(*, kid: int | None = None):
    arch = _runtime_arch()
    if arch not in _RAW_CASES:
        pytest.skip("requires a gfx942/gfx950/gfx1250 ROCm GPU")
    spec = dict(_RAW_CASES[arch])
    if kid is not None:
        spec["kid"] = kid
    device = torch.device("cuda", torch.cuda.current_device())
    config = select_launch_config(
        arch=arch,
        M=spec["M"],
        N=spec["N"],
        K=spec["K"],
        batch=1,
        cu_num=torch.cuda.get_device_properties(device).multi_processor_count,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        explicit_kid=spec["kid"],
        explicit_split_k=spec["split_k"],
        tuned_lookup=lambda **_kwargs: None,
    )
    instance = _instance(arch, config.actual_kid)
    plan = plan_a16w16_workspace(
        instance,
        arch=arch,
        kid=config.actual_kid,
        M=spec["M"],
        N=spec["N"],
        K=spec["K"],
        batch=1,
        split_k=config.allocation_split_k,
    )
    assert plan is not None
    XQ = torch.randn(
        (1, spec["M"], spec["K"]), device=device, dtype=torch.bfloat16
    )
    WQ = torch.randn(
        (1, spec["N"], spec["K"]), device=device, dtype=torch.bfloat16
    )
    Y = torch.empty(
        (1, spec["M"], spec["N"]), device=device, dtype=torch.bfloat16
    )
    return config, plan, XQ, WQ, Y


def _raw_launch(case, workspace):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config, _plan, XQ, WQ, Y = case
    return gemm._opus_gemm_a16w16_tune_raw(
        XQ,
        WQ,
        Y,
        None,
        workspace,
        config.actual_kid,
        config.launch_split_k,
    )


@pytest.mark.parametrize(
    ("arch", "kid", "expected_dtype"),
    [
        ("gfx950", 200, torch.float32),
        ("gfx942", 10200, torch.float32),
        ("gfx942", 10210, torch.bfloat16),
        ("gfx1250", 20000, torch.float32),
    ],
)
def test_raw_cpp_accepts_exact_typed_workspace(arch, kid, expected_dtype):
    if _runtime_arch() != arch:
        pytest.skip(f"requires {arch} hardware")
    case = _make_raw_case(kid=kid)
    _config, plan, _XQ, _WQ, Y = case
    assert plan.dtype == expected_dtype
    workspace = torch.empty(
        plan.required_numel, device=Y.device, dtype=expected_dtype
    )
    _raw_launch(case, workspace)
    torch.cuda.synchronize(Y.device)
    assert torch.isfinite(Y).all()


def test_raw_cpp_rejects_workspace_one_element_short():
    case = _make_raw_case()
    _config, plan, _XQ, _WQ, Y = case
    workspace = torch.empty(
        plan.required_numel - 1, device=Y.device, dtype=plan.dtype
    )
    with pytest.raises(RuntimeError, match="workspace capacity.*elements"):
        _raw_launch(case, workspace)


@pytest.mark.parametrize("failure", ["missing", "dtype", "noncontiguous", "alignment"])
def test_raw_cpp_rejects_invalid_workspace_contract(failure):
    case = _make_raw_case()
    _config, plan, _XQ, _WQ, Y = case
    if failure == "missing":
        workspace = None
        message = "requires a workspace tensor"
    elif failure == "dtype":
        wrong_dtype = (
            torch.bfloat16 if plan.dtype == torch.float32 else torch.float32
        )
        workspace = torch.empty(
            plan.required_numel, device=Y.device, dtype=wrong_dtype
        )
        message = "workspace dtype must be"
    elif failure == "noncontiguous":
        workspace = torch.empty(
            (plan.required_numel, 2), device=Y.device, dtype=plan.dtype
        )[:, 0]
        assert not workspace.is_contiguous()
        message = "workspace must be contiguous"
    else:
        workspace = torch.empty(
            plan.required_numel + 1, device=Y.device, dtype=plan.dtype
        )[1:]
        assert workspace.is_contiguous()
        assert workspace.data_ptr() % plan.alignment != 0
        message = "workspace address must be aligned"

    with pytest.raises(RuntimeError, match=message):
        _raw_launch(case, workspace)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="requires two ROCm devices for the C++ device-id guard",
)
def test_raw_cpp_rejects_workspace_on_another_device():
    case = _make_raw_case()
    _config, plan, XQ, _WQ, _Y = case
    input_index = XQ.device.index
    other_index = (input_index + 1) % torch.cuda.device_count()
    workspace = torch.empty(
        plan.required_numel,
        device=torch.device("cuda", other_index),
        dtype=plan.dtype,
    )
    with pytest.raises(RuntimeError, match="workspace device.*must match input device"):
        _raw_launch(case, workspace)


def test_raw_cpp_non_workspace_kid_requires_none():
    if _runtime_arch() != "gfx950":
        pytest.skip("the checked non-workspace fixture is gfx950-specific")
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    device = torch.device("cuda", torch.cuda.current_device())
    XQ = torch.empty((1, 192, 128), device=device, dtype=torch.bfloat16)
    WQ = torch.empty((1, 64, 128), device=device, dtype=torch.bfloat16)
    Y = torch.empty((1, 192, 64), device=device, dtype=torch.bfloat16)
    workspace = torch.empty(1, device=device, dtype=torch.float32)
    with pytest.raises(
        RuntimeError, match="non-workspace kernel id 300.*workspace=None"
    ):
        gemm._opus_gemm_a16w16_tune_raw(
            XQ, WQ, Y, None, workspace, 300, 0
        )


def test_gfx1250_raw_cpp_rejects_batch_greater_than_one():
    if _runtime_arch() != "gfx1250":
        pytest.skip("requires gfx1250 hardware")
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    device = torch.device("cuda", torch.cuda.current_device())
    XQ = torch.empty((2, 16, 512), device=device, dtype=torch.bfloat16)
    WQ = torch.empty((2, 32, 512), device=device, dtype=torch.bfloat16)
    Y = torch.empty((2, 16, 32), device=device, dtype=torch.bfloat16)
    workspace = torch.empty(2048, device=device, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="supports batch == 1 only"):
        gemm._opus_gemm_a16w16_tune_raw(
            XQ, WQ, Y, None, workspace, 20000, 2
        )
