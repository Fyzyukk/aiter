# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-side parity tests for the OPUS a16w16 Python selector."""

from __future__ import annotations

import importlib

import pytest
import torch

from csrc.opus_gemm.opus_gemm_common import (
    GFX942_BF16WS_EXACT_N,
    get_kernel_instance,
    kernel_needs_external_workspace,
)

from aiter.ops.opus._selector_a16w16 import LaunchConfig, select_launch_config
from aiter.ops.opus.heuristics.a16w16_gfx1250 import select_kid as select_gfx1250
from aiter.ops.opus.heuristics.a16w16_gfx942 import (
    launcher_symbol_to_kid,
    resolve_split_k,
    select_kid as select_gfx942,
)
from aiter.ops.opus.heuristics.a16w16_gfx950 import select_kid as select_gfx950


def _csv_miss(**_kwargs):
    return None


def _select(
    arch: str,
    M: int,
    N: int,
    K: int,
    *,
    batch: int = 1,
    cu_num: int = 304,
    has_bias: bool = False,
    output_dtype=torch.bfloat16,
    explicit_kid: int | None = None,
    explicit_split_k: int | None = None,
    tuned_lookup=_csv_miss,
) -> LaunchConfig:
    return select_launch_config(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=batch,
        cu_num=cu_num,
        has_bias=has_bias,
        input_dtype=torch.bfloat16,
        output_dtype=output_dtype,
        explicit_kid=explicit_kid,
        explicit_split_k=explicit_split_k,
        tuned_lookup=tuned_lookup,
    )


def test_common_queries_are_arch_and_family_scoped():
    instance = get_kernel_instance("gfx942", "a16w16", 10200)
    assert instance is not None
    assert instance.name.startswith("opus_gemm_gfx942_splitk_legacy_")
    assert get_kernel_instance("gfx950", "a16w16", 10200) is None
    assert get_kernel_instance("gfx942", "a8w8", 11000) is None
    assert get_kernel_instance("gfx942", "a16w16", 11000) is None

    assert kernel_needs_external_workspace("gfx942", "a16w16", 10200)
    assert not kernel_needs_external_workspace("gfx950", "a16w16", 300)
    with pytest.raises(KeyError, match="unknown OPUS kernel"):
        kernel_needs_external_workspace("gfx942", "a16w16", 11000)


def test_gfx942_exact_n_is_shared_and_contains_384():
    assert GFX942_BF16WS_EXACT_N == frozenset(
        {64, 128, 256, 384, 512, 1024, 2048}
    )


@pytest.mark.parametrize(
    ("M", "N", "K", "expected"),
    [
        (32, 128, 4096, 20007),
        (32, 64, 4096, 20006),
        (32, 32, 4096, 20005),
        (17, 128, 4096, 20004),
        (17, 64, 4096, 20003),
        (17, 33, 4096, 20000),
    ],
)
def test_gfx1250_heuristic_matches_cpp_branches(M, N, K, expected):
    assert select_gfx1250(M, N, K) == expected


@pytest.mark.parametrize(
    ("M", "N", "K", "has_bias", "expected"),
    [
        (4, 64, 128, False, 208),
        (64, 32, 128, False, 1206),
        (63, 32, 128, False, 206),
        (128, 64, 64, False, 1200),
        (127, 65, 64, False, 200),
        (256, 256, 128, False, 1300),
        (192, 240, 128, False, 300),
        (256, 256, 128, True, 1200),
        (257, 257, 128, True, 200),
    ],
)
def test_gfx950_heuristic_matches_cpp_branches(M, N, K, has_bias, expected):
    assert select_gfx950(M, N, K, has_bias=has_bias) == expected


@pytest.mark.parametrize(
    ("M", "N", "K", "output_dtype", "has_bias", "expected"),
    [
        # Exact K=4096 DSV4 branches.
        (48, 1024, 4096, "bf16", False, 10213),
        (512, 256, 4096, "bf16", False, 10203),
        (64, 1536, 4096, "bf16", False, 10205),
        # WKC small-M/N branches.
        (4, 4096, 1024, "bf16", False, 10300),
        (16, 1536, 4096, "bf16", False, 10305),
        (32, 1536, 2048, "bf16", False, 10303),
        (128, 512, 2048, "bf16", False, 10302),
        # bf16-workspace and N=384 bands.
        (256, 1024, 7168, "bf16", False, 10210),
        (160, 384, 4096, "bf16", False, 10201),
        (400, 384, 4096, "bf16", False, 10204),
        # Large-N direct and split-barrier fallbacks.
        (128, 4096, 2048, "bf16", False, 10000),
        (128, 2048, 512, "bf16", False, 10000),
        (700, 256, 1024, "bf16", False, 10000),
        (700, 250, 1024, "bf16", False, 10201),
        (700, 512, 1000, "bf16", False, 10200),
        # fp32 output or bias takes the C++ non-bf16-specialized arm.
        (32, 256, 1024, "fp32", False, 10201),
        (32, 512, 1024, "fp32", False, 10200),
        (32, 256, 1024, "bf16", True, 10201),
    ],
)
def test_gfx942_heuristic_matches_cpp_branches(
    M, N, K, output_dtype, has_bias, expected
):
    assert (
        select_gfx942(
            M,
            N,
            K,
            has_bias=has_bias,
            output_dtype=output_dtype,
        )
        == expected
    )


def test_gfx942_launcher_symbol_is_resolved_from_instance_name():
    instance = get_kernel_instance("gfx942", "a16w16", 10204)
    assert instance is not None
    assert launcher_symbol_to_kid(instance.name) == 10204
    with pytest.raises(RuntimeError, match="unknown launcher symbol"):
        launcher_symbol_to_kid("not_an_opus_launcher")


def test_gfx942_auto_split_k_and_even_loop_down_clamp_match_launcher():
    instance = get_kernel_instance("gfx942", "a16w16", 10201)
    assert instance is not None
    resolution = resolve_split_k(
        instance,
        M=512,
        N=512,
        K=2048,
        batch=1,
        cu_num=304,
        requested=0,
    )
    assert (resolution.requested, resolution.allocation, resolution.effective) == (
        0,
        5,
        4,
    )


def test_gfx942_explicit_split_k_down_clamps_without_auto_ceiling():
    instance = get_kernel_instance("gfx942", "a16w16", 10200)
    assert instance is not None
    resolution = resolve_split_k(
        instance,
        M=128,
        N=128,
        K=4096,
        batch=1,
        cu_num=304,
        requested=17,
    )
    assert resolution.allocation == 17
    assert resolution.effective == 16


def test_gfx942_split_k_rejects_too_few_k_iterations():
    instance = get_kernel_instance("gfx942", "a16w16", 10201)
    assert instance is not None
    with pytest.raises(ValueError, match="too small"):
        resolve_split_k(
            instance,
            M=64,
            N=64,
            K=64,
            batch=1,
            cu_num=304,
            requested=1,
        )


def test_explicit_selection_precedes_tuned_lookup():
    def must_not_run(**_kwargs):
        raise AssertionError("tuned lookup ran before explicit selection")

    config = _select(
        "gfx950",
        128,
        64,
        4096,
        explicit_kid=200,
        explicit_split_k=3,
        tuned_lookup=must_not_run,
    )
    assert config.source == "explicit"
    assert (config.requested_kid, config.actual_kid) == (200, 200)
    assert (config.requested_split_k, config.allocation_split_k) == (3, 3)


def test_explicit_unknown_or_wrong_arch_kid_fails_strictly():
    with pytest.raises(ValueError, match="not an a16w16 kernel for gfx942"):
        _select("gfx942", 128, 128, 4096, explicit_kid=200)


def test_valid_tuned_row_preserves_its_kid_split_pair():
    config = _select(
        "gfx950",
        128,
        64,
        4096,
        tuned_lookup=lambda **_kwargs: {"solidx": 200, "splitK": 7},
    )
    assert config.source == "tuned"
    assert (config.requested_kid, config.actual_kid) == (200, 200)
    assert (config.requested_split_k, config.launch_split_k) == (7, 7)


def test_wrong_arch_tuned_row_discards_kid_and_split_k_atomically():
    config = _select(
        "gfx942",
        700,
        250,
        1024,
        tuned_lookup=lambda **_kwargs: {"solidx": 200, "splitK": 9},
    )
    assert config.source == "heuristic"
    assert config.requested_kid == 10201
    assert config.requested_split_k == 0
    assert config.allocation_split_k != 9


def test_shape_invalid_tuned_mono_row_falls_back_as_a_pair():
    config = _select(
        "gfx950",
        64,
        128,
        128,
        tuned_lookup=lambda **_kwargs: {"solidx": 1400, "splitK": 11},
    )
    assert config.source == "heuristic"
    assert config.requested_kid == 1206
    assert config.requested_split_k == 0


@pytest.mark.parametrize(
    ("requested", "expected"),
    [(10210, 10200), (10213, 10203)],
)
def test_gfx942_non_exact_n_redirects_before_split_resolution(requested, expected):
    config = _select(
        "gfx942",
        128,
        768,
        4096,
        explicit_kid=requested,
        explicit_split_k=0,
    )
    assert config.requested_kid == requested
    assert config.actual_kid == expected
    assert config.effective_split_k is not None


def test_gfx942_10216_rejects_non_exact_n():
    with pytest.raises(ValueError, match="kid 10216 requires exact-N"):
        _select(
            "gfx942",
            256,
            768,
            4096,
            explicit_kid=10216,
        )


@pytest.mark.parametrize("requested", [10210, 10213, 10216])
def test_gfx942_n384_keeps_bf16_workspace_launcher(requested):
    config = _select(
        "gfx942",
        256,
        384,
        4096,
        explicit_kid=requested,
    )
    assert config.actual_kid == requested


def test_invalid_tuned_10216_discards_split_then_runs_heuristic():
    config = _select(
        "gfx942",
        256,
        768,
        4096,
        tuned_lookup=lambda **_kwargs: {"solidx": 10216, "splitK": 13},
    )
    assert config.source == "heuristic"
    assert config.requested_kid == 10210
    assert config.actual_kid == 10200
    assert config.requested_split_k == 0


def test_unsupported_arch_reaches_framework_fallback():
    config = _select("gfx1100", 64, 64, 128)
    assert config.is_framework_fallback
    assert config.source == "framework"
    assert config.requested_kid is None
    assert config.actual_kid is None


def test_heuristic_kid_must_be_in_force_compiled_set(monkeypatch):
    selector = importlib.import_module("aiter.ops.opus._selector_a16w16")
    monkeypatch.setitem(
        selector.A16W16_HEURISTICS,
        "gfx950",
        lambda *_args, **_kwargs: 999999,
    )
    with pytest.raises(RuntimeError, match="HEURISTIC_DEFAULT_KIDS_BY_ARCH"):
        _select("gfx950", 64, 64, 128)


def test_gfx1250_batched_request_is_rejected_before_launch():
    with pytest.raises(ValueError, match="requires batch=1"):
        _select("gfx1250", 32, 128, 4096, batch=2, cu_num=256)


def test_gfx942_bias_respects_current_tune_abi_gate():
    config = _select("gfx942", 128, 256, 4096, has_bias=True)
    assert config.is_framework_fallback
    assert "rejects bias" in config.fallback_reason


def test_csv_miss_production_path_uses_tune_wrapper_not_generic_cpp(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    common = importlib.import_module("aiter.ops.opus.common")
    monkeypatch.setattr(gemm, "_device_arch_and_cu", lambda _device: ("gfx950", 256))
    monkeypatch.setattr(common, "lookup_tuned", _csv_miss)

    calls = []

    def fake_tune(XQ, WQ, Y, bias, kernelId, splitK):
        calls.append((tuple(XQ.shape), tuple(WQ.shape), kernelId, splitK, bias))
        Y.zero_()
        return Y

    def generic_cpp_must_not_run(*_args, **_kwargs):
        raise AssertionError("CSV miss called the legacy generic C++ selector")

    monkeypatch.setattr(gemm, "opus_gemm_a16w16_tune", fake_tune)
    monkeypatch.setattr(gemm, "_opus_gemm_bf16_dispatch", generic_cpp_must_not_run)

    A = torch.empty((64, 128), dtype=torch.bfloat16)
    B = torch.empty((32, 128), dtype=torch.bfloat16)
    output = gemm.gemm_a16w16_opus(A, B)

    assert output.shape == (64, 32)
    assert calls == [((1, 64, 128), (1, 32, 128), 1206, 0, None)]
