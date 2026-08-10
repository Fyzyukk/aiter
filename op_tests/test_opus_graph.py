# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Graph, stream-ownership, and lifetime regressions for OPUS workspaces."""

from __future__ import annotations

import gc
import importlib
import weakref

import pytest
import torch

from aiter.ops.opus._selector_a16w16 import LaunchConfig


_GRAPH_CASES = {
    "gfx950": dict(kid=200, M=64, N=64, K=512, split_k=2),
    "gfx942": dict(kid=10200, M=128, N=128, K=512, split_k=2),
    "gfx1250": dict(kid=20000, M=16, N=32, K=512, split_k=2),
}


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()


def _require_graph_case(arch: str):
    if _runtime_arch() != arch:
        pytest.skip(f"requires {arch} hardware")
    return dict(_GRAPH_CASES[arch])


def _load_raw_binding_without_workspace_launch(gemm) -> None:
    """Load the JIT module before capture without prewarming any workspace."""
    device = torch.device("cuda", torch.cuda.current_device())
    XQ = torch.empty((1, 1, 2), device=device, dtype=torch.bfloat16)
    WQ = torch.empty((1, 1, 2), device=device, dtype=torch.bfloat16)
    Y = torch.empty((1, 1, 1), device=device, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="Kernel id -999"):
        gemm._opus_gemm_a16w16_tune_raw(XQ, WQ, Y, None, None, -999, 0)


def _golden(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A.float() @ B.float().transpose(-1, -2)


@pytest.mark.parametrize("arch", ["gfx950", "gfx942", "gfx1250"])
def test_graph_capture_replay_allocates_in_capture_without_prewarm(monkeypatch, arch):
    spec = _require_graph_case(arch)
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    _load_raw_binding_without_workspace_launch(gemm)

    def forbidden_prewarm():
        raise AssertionError("graph path called the deprecated workspace prewarm")

    monkeypatch.setattr(gemm, "opus_gemm_workspace_init", forbidden_prewarm)
    real_allocate = gemm.allocate_workspace
    allocation_ptrs = []

    def record_allocate(plan, device):
        workspace = real_allocate(plan, device)
        allocation_ptrs.append(workspace.data_ptr())
        return workspace

    monkeypatch.setattr(gemm, "allocate_workspace", record_allocate)
    A = torch.randn((spec["M"], spec["K"]), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((spec["N"], spec["K"]), device="cuda", dtype=torch.bfloat16)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gemm.gemm_a16w16_opus(
            A,
            B,
            dtype=torch.bfloat16,
            kernelId=spec["kid"],
            splitK=spec["split_k"],
        )

    # The target shape was never launched eagerly. Its single workspace was
    # allocated by torch.empty while capture was active; replay is pure graph.
    assert len(allocation_ptrs) == 1
    for seed in (7, 11, 19):
        generator = torch.Generator(device=A.device).manual_seed(seed)
        A.copy_(
            torch.randn(A.shape, device=A.device, dtype=A.dtype, generator=generator)
        )
        B.copy_(
            torch.randn(B.shape, device=B.device, dtype=B.dtype, generator=generator)
        )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.float(), _golden(A, B), rtol=0.03, atol=0.5
        )
        assert len(allocation_ptrs) == 1


@pytest.mark.parametrize("arch", ["gfx950", "gfx942", "gfx1250"])
def test_two_streams_hold_distinct_call_scoped_workspaces(monkeypatch, arch):
    spec = _require_graph_case(arch)
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    _load_raw_binding_without_workspace_launch(gemm)
    real_raw = gemm._opus_gemm_a16w16_tune_raw
    held_workspaces = []

    def record_raw(XQ, WQ, Y, bias, workspace, kid, split_k):
        held_workspaces.append(workspace)
        return real_raw(XQ, WQ, Y, bias, workspace, kid, split_k)

    monkeypatch.setattr(gemm, "_opus_gemm_a16w16_tune_raw", record_raw)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    inputs = [
        (
            torch.randn(
                (spec["M"], spec["K"]), device="cuda", dtype=torch.bfloat16
            ),
            torch.randn(
                (spec["N"], spec["K"]), device="cuda", dtype=torch.bfloat16
            ),
        )
        for _ in streams
    ]
    outputs = []
    producer = torch.cuda.current_stream()
    for stream, (A, B) in zip(streams, inputs, strict=True):
        stream.wait_stream(producer)
        with torch.cuda.stream(stream):
            outputs.append(
                gemm.gemm_a16w16_opus(
                    A,
                    B,
                    dtype=torch.bfloat16,
                    kernelId=spec["kid"],
                    splitK=spec["split_k"],
                )
            )
    for stream in streams:
        producer.wait_stream(stream)
    torch.cuda.synchronize()

    assert len(held_workspaces) == 2
    assert held_workspaces[0] is not held_workspaces[1]
    assert held_workspaces[0].data_ptr() != held_workspaces[1].data_ptr()
    for output, (A, B) in zip(outputs, inputs, strict=True):
        torch.testing.assert_close(
            output.float(), _golden(A, B), rtol=0.03, atol=0.5
        )


def test_many_shapes_leave_no_python_workspace_tensor_cache():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    dead_refs = []

    for M, N in ((33, 17), (64, 64), (65, 97), (129, 33)):
        config = LaunchConfig(
            arch="gfx950",
            family="a16w16",
            source="explicit",
            requested_kid=200,
            actual_kid=200,
            requested_split_k=2,
            allocation_split_k=2,
            launch_split_k=2,
        )
        XQ = torch.empty((1, M, 512), dtype=torch.bfloat16)
        WQ = torch.empty((1, N, 512), dtype=torch.bfloat16)
        Y = torch.empty((1, M, N), dtype=torch.bfloat16)

        def fake_raw(_XQ, _WQ, _Y, _bias, workspace, _kid, _split_k):
            dead_refs.append(weakref.ref(workspace))

        gemm._launch_a16w16_with_torch_workspace(
            fake_raw, XQ, WQ, Y, None, config
        )

    del XQ, WQ, Y
    gc.collect()
    assert dead_refs and all(reference() is None for reference in dead_refs)

    workspace_module = importlib.import_module("aiter.ops.opus._workspace")
    planner_module = importlib.import_module("aiter.ops.opus._workspace_a16w16")
    for module in (gemm, workspace_module, planner_module):
        cached = [
            name
            for name, value in vars(module).items()
            if isinstance(value, torch.Tensor)
        ]
        assert cached == [], f"{module.__name__} caches Tensor globals: {cached}"


def test_deprecated_workspace_init_is_a_warning_only_noop():
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    with pytest.warns(DeprecationWarning, match="deprecated and no longer required"):
        assert gemm.opus_gemm_workspace_init() is None
