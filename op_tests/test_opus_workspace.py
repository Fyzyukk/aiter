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


def test_step2_production_path_does_not_enable_torch_workspace_yet(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    common = importlib.import_module("aiter.ops.opus.common")
    monkeypatch.setattr(gemm, "_device_arch_and_cu", lambda _device: ("gfx950", 256))
    monkeypatch.setattr(common, "lookup_tuned", lambda **_kwargs: None)

    def must_not_allocate(*_args, **_kwargs):
        raise AssertionError("Step 2 enabled Torch workspace before the raw ABI")

    calls = []

    def fake_tune(XQ, WQ, Y, bias, kernelId, splitK):
        calls.append((kernelId, splitK))
        Y.zero_()
        return Y

    monkeypatch.setattr(gemm, "allocate_workspace", must_not_allocate)
    monkeypatch.setattr(gemm, "opus_gemm_a16w16_tune", fake_tune)

    output = gemm.gemm_a16w16_opus(
        torch.empty((64, 128), dtype=torch.bfloat16),
        torch.empty((32, 128), dtype=torch.bfloat16),
    )
    assert output.shape == (64, 32)
    assert calls == [(1206, 0)]


def test_non_workspace_kid_rejects_explicit_workspace():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    config = _workspace_config(kid=300, allocation_split_k=1, launch_split_k=0)
    XQ = torch.empty((1, 256, 128), dtype=torch.bfloat16)
    Y = torch.empty((1, 256, 256), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="does not use an external workspace"):
        gemm._prepare_a16w16_workspace(
            config, XQ, Y, torch.empty(1, dtype=torch.float32)
        )
