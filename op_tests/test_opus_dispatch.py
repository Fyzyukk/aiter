# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-side policy tests for the OPUS a16w16 Python selector."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

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

_ROOT = Path(__file__).resolve().parents[1]


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


def _gfx950_policy_golden(M, N, K, has_bias):
    """Independent B5 golden for the Python-owned gfx950 policy."""
    split_barrier_ok = N % 16 == 0 and K % 64 == 0 and (K // 64) % 2 == 0
    if M <= 4:
        return 1208 if M % 64 == 0 and N % 64 == 0 and K % 128 == 0 else 208
    if M <= 64:
        return 1206 if M % 64 == 0 and N % 32 == 0 and K % 128 == 0 else 206
    if M <= 128:
        return 1200 if M % 64 == 0 and N % 64 == 0 and K % 64 == 0 else 200
    if split_barrier_ok and not has_bias:
        return 1300 if M % 256 == 0 and N % 256 == 0 else 300
    return 1200 if M % 64 == 0 and N % 64 == 0 and K % 64 == 0 else 200


def _gfx1250_policy_golden(M, N):
    """Independent B5 golden for the Python-owned gfx1250 policy."""
    if M % 32 == 0:
        if N % 128 == 0:
            return 20007
        if N % 64 == 0:
            return 20006
        if N % 32 == 0:
            return 20005
    if N % 128 == 0:
        return 20004
    if N % 64 == 0:
        return 20003
    return 20000


def _generated_launcher_gfx942_split_golden(
    instance, M, N, K, batch, cu_num, requested
):
    """Frozen generated-launcher split resolver, independent of Python port."""
    if requested > 0:
        allocation = requested
    else:
        tiles_mn = (
            ((M + instance.B_M - 1) // instance.B_M)
            * ((N + instance.B_N - 1) // instance.B_N)
            * batch
        )
        target_wg = 2 * cu_num if instance.kernel_tag.endswith("_p1") else cu_num
        allocation = min(16, max(1, (target_wg + tiles_mn - 1) // tiles_mn))

    total_iters = (K + instance.B_K - 1) // instance.B_K
    if total_iters < 2:
        raise ValueError("too small")
    even_tags = {
        "a16w16_kbuf2v_sk",
        "a16w16_kbuf2v_bk128_sk",
        "a16w16_quad_mfma32_kbuf1_sk",
    }
    require_even = instance.kernel_tag in even_tags
    effective = allocation
    while effective > 1:
        iters_full = (total_iters + effective - 1) // effective
        last_loops = total_iters - (effective - 1) * iters_full
        parity_ok = not require_even or (
            iters_full % 2 == 0 and last_loops % 2 == 0
        )
        if iters_full >= 2 and last_loops >= 2 and parity_ok:
            break
        effective -= 1
    if require_even:
        iters_full = (total_iters + effective - 1) // effective
        last_loops = total_iters - (effective - 1) * iters_full
        if iters_full % 2 or last_loops % 2:
            raise ValueError("even loops")
    return requested, allocation, effective


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
def test_gfx1250_python_heuristic_policy_golden(M, N, K, expected):
    assert select_gfx1250(M, N, K) == expected


def test_gfx1250_python_selector_policy_boundary_sweep():
    for M in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65):
        for N in (31, 32, 33, 63, 64, 65, 127, 128, 129, 256):
            expected = _gfx1250_policy_golden(M, N)
            assert select_gfx1250(M, N, 4096) == expected


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
def test_gfx950_python_heuristic_policy_golden(M, N, K, has_bias, expected):
    assert select_gfx950(M, N, K, has_bias=has_bias) == expected


def test_gfx950_python_selector_policy_boundary_sweep():
    for M in (1, 4, 5, 63, 64, 65, 127, 128, 129, 192, 255, 256, 257):
        for N in (31, 32, 63, 64, 65, 240, 256, 257):
            for K in (64, 127, 128, 192, 256):
                for has_bias in (False, True):
                    expected = _gfx950_policy_golden(M, N, K, has_bias)
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
        # fp32 output or bias takes the Python non-bf16 policy arm.
        (32, 256, 1024, "fp32", False, 10201),
        (32, 512, 1024, "fp32", False, 10200),
        (32, 256, 1024, "bf16", True, 10201),
    ],
)
def test_gfx942_python_heuristic_policy_golden(
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


def test_cpp_runtime_shape_policy_is_removed_but_physical_4g_safety_remains():
    generator = (_ROOT / "csrc/opus_gemm/gen_instances.py").read_text()
    common = (_ROOT / "csrc/opus_gemm/opus_gemm_common.py").read_text()

    assert "def gen_lookup_dict(" not in generator
    assert "def get_tune_dict(" not in generator
    assert "_combined_opus_tuned.csv" not in generator
    assert "codegen.gen_lookup_dict" not in generator
    assert "default_kernels_dict" not in common

    for arch in ("gfx942", "gfx950", "gfx1250"):
        include_dir = _ROOT / f"csrc/opus_gemm/include/{arch}"
        header = (include_dir / f"opus_gemm_arch_{arch}.cuh").read_text()
        assert '#include "opus_gemm_a16w16_kid_dispatch.h"' in header
        assert "find_kid(" in header
        assert "workspace_entry(" in header
        for removed in (
            "opus_gemm_lookup.h",
            "opus_gemm_a16w16_tune_lookup.h",
            "OpusA16W16Shape",
            "OpusA16W16RuntimeEntry",
            "find_shape_kid",
            "opus_select_a16w16_kid",
            "GENERATE_OPUS_LOOKUP_TABLE",
            "check_shape_4g",
        ):
            assert removed not in header
        assert not (include_dir / f"opus_gemm_heuristic_dispatch_{arch}.cuh").exists()

    tuner = (_ROOT / "csrc/opus_gemm/opus_gemm_tune.py").read_text()
    assert 'getattr(k_inst, "is_4g_safe", False)' in tuner
    assert "M * K * 2 > _UINT32_MAX_BYTES" in tuner
    safe_pipeline = (
        _ROOT
        / "csrc/opus_gemm/include/gfx950/"
        "opus_gemm_pipeline_a16w16_4g_safe_gfx950.cuh"
    ).read_text()
    assert "Per-WG-tight BR sizing" in safe_pipeline
    assert "(size_t)batch_id" in safe_pipeline


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


@pytest.mark.parametrize(
    "kid", [10200, 10201, 10203, 10204, 10210, 10213, 10216]
)
@pytest.mark.parametrize("requested", [0, 1, 3, 17])
def test_gfx942_split_resolver_matches_generated_cpp(kid, requested):
    instance = get_kernel_instance("gfx942", "a16w16", kid)
    assert instance is not None
    expected = _generated_launcher_gfx942_split_golden(
        instance, 257, 769, 4096, 2, 304, requested
    )
    actual = resolve_split_k(
        instance,
        M=257,
        N=769,
        K=4096,
        batch=2,
        cu_num=304,
        requested=requested,
    )
    assert (actual.requested, actual.allocation, actual.effective) == expected


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


def test_tuned_selection_precedes_heuristic(monkeypatch):
    selector = importlib.import_module("aiter.ops.opus._selector_a16w16")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("heuristic ran before a valid tuned selection")

    monkeypatch.setitem(selector.A16W16_HEURISTICS, "gfx950", must_not_run)
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
@pytest.mark.parametrize("N", [63, 65, 768, 2049])
def test_gfx942_non_exact_n_redirects_before_split_resolution(
    requested, expected, N
):
    config = _select(
        "gfx942",
        128,
        N,
        4096,
        explicit_kid=requested,
        explicit_split_k=0,
    )
    assert config.requested_kid == requested
    assert config.actual_kid == expected
    assert config.effective_split_k is not None


@pytest.mark.parametrize("N", [63, 65, 768, 2049])
def test_gfx942_10216_rejects_non_exact_n(N):
    with pytest.raises(ValueError, match="kid 10216 requires exact-N"):
        _select(
            "gfx942",
            256,
            N,
            4096,
            explicit_kid=10216,
        )


@pytest.mark.parametrize("requested", [10210, 10213, 10216])
@pytest.mark.parametrize("N", sorted(GFX942_BF16WS_EXACT_N))
def test_gfx942_all_exact_n_values_keep_bf16_workspace_launcher(requested, N):
    config = _select(
        "gfx942",
        256,
        N,
        4096,
        explicit_kid=requested,
    )
    assert config.actual_kid == requested


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


def test_framework_fallback_executes_without_touching_raw_binding(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    common = importlib.import_module("aiter.ops.opus.common")
    monkeypatch.setattr(gemm, "_device_arch_and_cu", lambda _device: ("gfx1100", 1))
    monkeypatch.setattr(common, "lookup_tuned", _csv_miss)

    def raw_must_not_run(*_args, **_kwargs):
        raise AssertionError("framework fallback entered the OPUS raw binding")

    monkeypatch.setattr(
        gemm, "_opus_gemm_a16w16_launch_ctypes_raw", raw_must_not_run
    )
    A = torch.arange(24, dtype=torch.bfloat16).reshape(3, 8) / 8
    B = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8) / 16
    bias = torch.arange(4, dtype=torch.float32) / 4

    actual = gemm.gemm_a16w16_opus(A, B, bias=bias, dtype=torch.float32)
    expected = A.float() @ B.float().T + bias
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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


def test_csv_miss_production_path_uses_canonical_raw_binding(monkeypatch):
    gemm = importlib.import_module("aiter.ops.opus.gemm_op_a16w16")
    common = importlib.import_module("aiter.ops.opus.common")
    monkeypatch.setattr(gemm, "_device_arch_and_cu", lambda _device: ("gfx950", 256))
    monkeypatch.setattr(common, "lookup_tuned", _csv_miss)

    calls = []

    def fake_raw(XQ, WQ, Y, bias, workspace, kernelId, splitK):
        calls.append(
            (
                tuple(XQ.shape),
                tuple(WQ.shape),
                workspace,
                kernelId,
                splitK,
                bias,
            )
        )
        Y.zero_()
        return Y

    monkeypatch.setattr(gemm, "_opus_gemm_a16w16_launch_ctypes_raw", fake_raw)

    A = torch.empty((65, 512), dtype=torch.bfloat16)
    B = torch.empty((33, 512), dtype=torch.bfloat16)
    output = gemm.gemm_a16w16_opus(A, B)

    assert output.shape == (65, 33)
    assert len(calls) == 1
    x_shape, w_shape, workspace, kid, split_k, bias = calls[0]
    assert (x_shape, w_shape) == ((1, 65, 512), (1, 33, 512))
    assert workspace.shape == (1, 1, 128, 64)
    assert workspace.dtype == torch.float32
    assert (kid, split_k, bias) == (200, 0, None)


def test_workspace_scope_preserves_a8w8_and_a8w4_public_apis():
    a8w8 = importlib.import_module("aiter.ops.opus.gemm_op_a8w8")
    stage1 = importlib.import_module("aiter.ops.opus.moe_stage1_a8w4")
    stage2 = importlib.import_module("aiter.ops.opus.moe_stage2_a8w4")

    assert tuple(
        inspect.signature(a8w8.opus_gemm_a8w8_blockscale_bpreshuffle_tune).parameters
    ) == ("XQ", "WQ", "x_scale", "w_scale", "Y", "kernelId")
    assert tuple(inspect.signature(stage1.opus_moe_stage1_a8w4_fwd).parameters) == (
        "hidden_states",
        "w1",
        "hidden_scale",
        "w1_scale",
        "sorted_token_ids",
        "sorted_expert_ids",
        "num_valid_ids",
        "topk",
        "inter_dim_pad",
        "block_m",
        "kernelName",
        "activation",
        "bias",
        "out",
        "out_scale",
        "output_sorted",
        "swiglu_limit",
        "situ_beta",
        "situ_linear_beta",
    )
    assert tuple(
        inspect.signature(stage2.opus_moe_stage2_a8w4_decode_fwd).parameters
    ) == (
        "inter_states",
        "w2",
        "a2_scale",
        "w2_scale",
        "sorted_token_ids",
        "sorted_weights",
        "sorted_expert_ids",
        "num_valid_ids",
        "block_m",
        "inter_dim_pad",
        "out",
        "kernel_id",
        "return_per_slot",
        "route_out_dtype",
        "token_num",
        "topk",
    )


def test_gfx942_direct_workspace_pointer_keeps_wave_uniformization():
    root = Path(__file__).resolve().parents[1]
    include = root / "csrc/opus_gemm/include/gfx942/a16w16"
    traits = (include / "opus_gemm_traits_a16w16.cuh").read_text()
    assert "opus_gfx942_uniform_ws_ptr(Ptr ptr_ws)" in traits
    assert traits.count("__builtin_amdgcn_readfirstlane") >= 2
    assert "void*       __restrict__ ptr_ws" in traits
    assert "opus_splitk_ws_handle" not in traits
    assert "opus_splitk_ws_ptr" not in traits

    main_pipelines = (
        "opus_gemm_pipeline_a16w16_em3en4_lds1_pgr2_sk.cuh",
        "opus_gemm_pipeline_a16w16_kbuf1.cuh",
        "opus_gemm_pipeline_a16w16_kbuf2v.cuh",
        "opus_gemm_pipeline_a16w16_kbuf2v_bk128.cuh",
        "opus_gemm_pipeline_a16w16_quad_mfma32_kbuf1.cuh",
    )
    for filename in main_pipelines:
        source = (include / filename).read_text()
        assert "opus_gfx942_uniform_ws_ptr<D_WS>(kargs.ptr_ws)" in source

    reduce = (include / "splitk_reduce_gfx942.cuh").read_text()
    assert reduce.count("opus_gfx942_uniform_ws_ptr<D_WS>(ws_ptr)") >= 2
    assert "const void*" in reduce
    assert "opus_splitk_ws_handle" not in reduce
