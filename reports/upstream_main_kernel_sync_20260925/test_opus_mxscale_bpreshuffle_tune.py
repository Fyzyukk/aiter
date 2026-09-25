# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

import aiter.ops.opus as opus_api
from aiter.ops.opus import launch_plan
from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
from csrc.opus_gemm.opus_gemm_common import (
    a8w8_mxscale_gemm_bpreshuffle_kernels_list,
    get_kernel_instance,
    kernels_list,
)


@pytest.fixture
def tuner(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    instance = tune.OpusMxscaleBpreshuffleTuner()
    monkeypatch.setattr(instance, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(instance, "get_cu_num", lambda: 256)
    return instance


def shape_row(**kwargs):
    return {"gfx": "gfx950", "cu_num": 256, "M": 256, "N": 256, "K": 256, **kwargs}


def saved_row(**kwargs):
    return shape_row(
        dtype="fp8",
        outdtype="bf16",
        scale_dtype="e8m0",
        libtype="opus",
        kernelId=9000,
        splitK=0,
        **kwargs,
    )


def test_import_does_not_parse_or_start_tuning():
    with (
        patch("argparse.ArgumentParser.parse_args") as parse,
        patch("aiter.utility.mp_tuner.mp_tuner") as sweep,
    ):
        importlib.reload(tune)
    parse.assert_not_called()
    sweep.assert_not_called()


@pytest.mark.parametrize(
    "shape", [(256, 256, 128), (1280, 768, 384), (2048, 6144, 7168)]
)
def test_candidates_cover_all_registered_implementations(shape):
    expected = set(a8w8_mxscale_gemm_bpreshuffle_kernels_list)
    kids = tune.candidate_kids_for_shape("gfx950", *shape)
    assert set(kids) == expected
    assert kids == [9000, 9010, 9011, 9012, 9020]
    assert a8w8_mxscale_gemm_bpreshuffle_kernels_list[9000].output_tiles_per_wg == 1
    assert kernels_list[9000] is a8w8_mxscale_gemm_bpreshuffle_kernels_list[9000]
    for removed_kid in (9001, 9002, 9003):
        assert removed_kid not in a8w8_mxscale_gemm_bpreshuffle_kernels_list
    assert len(
        {a8w8_mxscale_gemm_bpreshuffle_kernels_list[k].name for k in kids}
    ) == len(kids)
    for kid in kids:
        assert get_kernel_instance("gfx950", "a8w8_mxscale_bpreshuffle", kid) is None
        assert (
            get_kernel_instance("gfx950", "a8w8_blockscale_bpreshuffle", kid, "bf16")
            is kernels_list[kid]
        )
        assert (
            get_kernel_instance("gfx950", "a8w8_blockscale_bpreshuffle", kid, "fp32")
            is None
        )
        assert (
            launch_plan._A8W8_FAMILY_BY_TAG[kernels_list[kid].kernel_tag]
            == launch_plan._A8W8_BPRESHUFFLE_FAMILY
        )
    assert get_kernel_instance("gfx942", "a8w8_blockscale_bpreshuffle", 11000, "bf16")
    assert get_kernel_instance("gfx942", "a8w8_mxscale_bpreshuffle", 11000) is None


@pytest.mark.parametrize(
    "explicit_request", [False, True], ids=["default", "extra-4wave-kid"]
)
def test_bpreshuffle_codegen_emits_typed_instances_and_exact_tables(
    tmp_path, explicit_request
):
    root = Path('/root/workspace/aiter-opus-mxfp8-bpreshuffle')
    command = [
        sys.executable,
        str(root / "csrc/opus_gemm/gen_instances.py"),
        "--working_path",
        str(tmp_path),
    ]
    if explicit_request:
        command += ["--extra_kids", "9000", "9010", "9011", "9012", "9020"]
    completed = subprocess.run(
        command,
        cwd=root,
        env={
            **os.environ,
            "GPU_ARCHS": "gfx950",
            "CU_NUM": "256",
            "AITER_AOT_IMPORT": "1",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    compiled = json.loads((tmp_path / "compiled_kids.json").read_text())
    assert not {9001, 9002, 9003}.intersection(compiled)
    dispatch = (tmp_path / "opus_gemm_a8w8_kid_dispatch.h").read_text()
    private_dispatch = tmp_path / "opus_gemm_mxscale_bpreshuffle_tune_kid_dispatch.h"
    assert not private_dispatch.exists()
    if not explicit_request:
        assert 9000 not in compiled
        assert "{ 9000," not in dispatch
        return

    assert 9000 in compiled
    host = (tmp_path / "instances/all_instances_host_gfx950.cu").read_text()
    manifest = (tmp_path / "opus_gemm_manifest.h").read_text()
    assert (
        "#define "
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_BF16_SIZE 5"
        in dispatch
    )
    assert (
        "#define "
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_FP32_SIZE 0"
        in dispatch
    )
    for kid in (9000, 9010, 9011, 9012, 9020):
        name = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid].name
        assert name in manifest
        assert f"{{ {kid}, &{name}<bf16_t> }}" in dispatch
        assert f"{name}<bf16_t>" in host
        assert f"{{ {kid}, &{name}<fp32_t> }}" not in dispatch
        assert not (tmp_path / f"instances/{name}_Cfp32_t.device.cu").exists()
        if kid in (9010, 9011, 9012):
            instance = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
            source = (tmp_path / f"impl/{name}.cuh").read_text()
            assert f"bpreshuffle_{instance.B_M}x{instance.B_N}_gfx950.cuh" in source
            assert f"bpreshuffle_{instance.B_M}x{instance.B_N}_kernel" in source
            tile = f"{instance.B_M}x{instance.B_N}"
            assert (
                '#include "gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_'
                f'{tile}_gfx950.cuh"' in source
            )
            assert (
                f"using {name}_Traits = "
                f"opus_gemm_mxscale_bpreshuffle_{tile}_traits_gfx950;"
            ) in source
            assert "EARLY_PREFETCH" not in source
            assert "early_prefetch" not in source
    four_wave_name = a8w8_mxscale_gemm_bpreshuffle_kernels_list[9000].name
    four_wave = (tmp_path / f"impl/{four_wave_name}.cuh").read_text()
    assert "gemm_a8w8_mxfp8_scale_kernel" in four_wave
    assert "blockscale_generic" not in four_wave
    assert "if (args.k == " not in four_wave
    assert "blockscale_panel::gemm_a8w8_mxfp8_scale_kernel" not in four_wave
    assert f"{four_wave_name}_WholeKTraits" not in four_wave
    assert (
        f"using {four_wave_name}_Traits = "
        "opus_gemm_mxscale_bpreshuffle_4wave_traits_gfx950;"
    ) in four_wave
    assert "const dim3 grid(n / 256, tiles_m);" in four_wave
    device = (tmp_path / f"instances/{four_wave_name}_Cbf16_t.device.cu").read_text()
    assert f"{four_wave_name}_Traits>" in device
    assert "gemm_a8w8_mxfp8_scale_kernel" in device
    assert "blockscale_generic" not in device
    assert "blockscale_panel::gemm_a8w8_mxfp8_scale_kernel" not in device
    padded = kernels_list[9020]
    padded_source = (tmp_path / f"impl/{padded.name}.cuh").read_text()
    assert "opus_gemm_mxscale_bpreshuffle_padded_m_traits_gfx950" in padded_source
    assert "const int tiles_m = (m + 255) / 256;" in padded_source
    assert "m % 64 == 0" in padded_source
    assert padded_source.count("<<<") == 1


def test_tuner_reuses_existing_jit_module_and_public_entry():
    root = Path('/root/workspace/aiter-opus-mxfp8-bpreshuffle')
    config = json.loads((root / "aiter/jit/optCompilerConfig.json").read_text())
    assert "module_deepgemm_opus_mxfp8_tune" not in config
    module = config["module_deepgemm_opus"]
    sources = "\n".join(module["srcs"])
    assert "/pybind/opus_gemm_pybind.cu" in sources
    assert "/opus_gemm/opus_gemm.cu" in sources
    assert "mxscale_bpreshuffle_tune" not in sources
    assert opus_api.__all__ == ["gemm_a16w16_opus", "opus_bmm", "opus_gemm"]
    assert not (root / "csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.cu").exists()
    assert not (
        root / "csrc/opus_gemm/include/opus_gemm_mxscale_bpreshuffle_tune.h"
    ).exists()
    assert not (
        root / "csrc/pybind/opus_gemm_mxscale_bpreshuffle_tune_pybind.cu"
    ).exists()


def test_subset_builder_reuses_existing_helper_and_restores_compiler_env(
    tmp_path, monkeypatch
):
    compiler = tmp_path / "patched-clang/bin"
    compiler.mkdir(parents=True)
    calls = []
    helper = types.ModuleType("opus_gemm_tune")

    def ensure(kids):
        calls.append((set(kids), os.environ.get("HIP_CLANG_PATH")))
        return True

    helper._ensure_kids_compiled = ensure
    monkeypatch.setitem(sys.modules, "opus_gemm_tune", helper)
    monkeypatch.setenv("OPUS_HIP_CLANG_PATH", str(compiler))
    monkeypatch.setenv("HIP_CLANG_PATH", "/previous/compiler")

    assert tune._ensure_kids_compiled({9000}) is True
    assert calls == [({9000}, str(compiler))]
    assert os.environ["HIP_CLANG_PATH"] == "/previous/compiler"


@pytest.mark.parametrize(
    "shape",
    [
        (0, 256, 128),
        (255, 256, 128),
        (256, 255, 128),
        (256, 256, 127),
        (65536, 16384, 1536),
        (16384, 65536, 1536),
        (65536, 65536, 1536),
    ],
)
def test_tuner_excludes_unsupported_tiles_and_large_addressing(shape):
    assert tune.candidate_kids_for_shape("gfx950", *shape) == []


def test_arch_and_output_membership():
    for gfx in ("gfx942", "gfx1250"):
        assert tune.candidate_kids_for_shape(gfx, 256, 256, 256) == []
    for outdtype in ("fp16", "fp32"):
        assert tune.candidate_kids_for_shape("gfx950", 256, 256, 256, outdtype) == []


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((128, 128, 128), [9010, 9011, 9012]),
        ((64, 128, 256), [9011, 9012]),
        ((192, 384, 8320), [9011, 9012]),
        ((64, 64, 128), []),  # Public compact-scale ABI retains N128 groups.
        ((63, 128, 128), []),
    ],
)
def test_small_tile_candidate_alignment(shape, expected):
    assert tune.candidate_kids_for_shape("gfx950", *shape) == expected


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((1088, 768, 7168), [9011, 9012, 9020]),
        ((1152, 2048, 7168), [9010, 9011, 9012, 9020]),
        ((64, 256, 128), [9011, 9012, 9020]),
        ((192, 512, 8320), [9011, 9012, 9020]),
        ((63, 256, 128), []),
    ],
)
def test_padded_large_tile_competes_with_small_tiles(shape, expected):
    assert tune.candidate_kids_for_shape("gfx950", *shape) == expected
    assert kernels_list[9020].m_align == 64
    assert kernels_list[9020].pad_m


def test_native_e8m0_data_and_standard_preshuffle():
    data = tune.generate_data(256, 512, 256, 9000, device="cpu")
    sa, sb = data["x_scale"], data["w_scale"]
    assert data["x"].dtype == data["w"].dtype == torch.float8_e4m3fn
    assert sa.dtype == sb.dtype == torch.float8_e8m0fnu
    assert sa.shape == (256, 2) and sa.stride() == (1, 256)
    assert sb.shape == (4, 2) and sb.is_contiguous()
    assert torch.unique(sa.view(torch.uint8)).numel() > 1
    assert torch.unique(sb.view(torch.uint8)).numel() > 1
    torch.testing.assert_close(data["x_scale_fp32"], sa.float())
    torch.testing.assert_close(data["w_scale_fp32"], sb.float())
    torch.testing.assert_close(
        data["x_scale_t_fp32"].view(-1),
        data["x_scale_fp32"].T.contiguous().view(-1),
    )
    assert data["weight_shuffle"].data_ptr() == data["w"].data_ptr()
    assert data["weight"].data_ptr() == data["w_reference"].data_ptr()
    # Independent byte-address checks for the established (16,16) B layout.
    packed = data["w"].view(torch.uint8).flatten()
    original = data["w_reference"].view(torch.uint8)
    for n in (0, 1, 15, 16, 255, 511):
        for k in (0, 15, 16, 31, 32, 127, 255):
            offset = (
                (((n // 16) * 8 + k // 32) * 2 + (k % 32) // 16) * 256
                + (n % 16) * 16
                + k % 16
            )
            assert packed[offset] == original[n, k]


def test_reference_respects_independent_m_n_and_k_scales():
    x = torch.ones((2, 256)).to(torch.float8_e4m3fn)
    w = torch.ones((256, 256)).to(x.dtype)
    sa = torch.tensor([[1.0, 2.0], [4.0, 8.0]]).to(torch.float8_e8m0fnu)
    sb = torch.tensor([[0.5, 1.0], [2.0, 4.0]]).to(torch.float8_e8m0fnu)
    expected = torch.tensor([[320.0, 1280.0], [1280.0, 5120.0]]).repeat_interleave(
        128, 1
    )
    torch.testing.assert_close(tune.run_torch(x, w, sa, sb), expected)


def test_reference_magnitude_preserves_cancelling_products():
    x = (
        torch.cat((torch.ones(128), -torch.ones(128)))
        .view(1, 256)
        .to(torch.float8_e4m3fn)
    )
    w = torch.ones((128, 256)).to(x.dtype)
    scales = torch.ones((1, 2)).to(torch.float8_e8m0fnu)
    expected, magnitude = tune.run_torch(x, w, scales, scales, with_bounds=True)
    torch.testing.assert_close(expected, torch.zeros((1, 128)))
    torch.testing.assert_close(magnitude, torch.full((1, 128), 256.0))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_accumulation_bound_accepts_cancellation_but_rejects_outliers(dtype):
    ref = torch.tensor([[[0.0]], [[256.0]]])
    assert tune.compare_outputs(ref, torch.tensor([[0.012]], dtype=dtype)) == 0
    assert tune.compare_outputs(ref, torch.tensor([[0.02]], dtype=dtype)) == 1


def test_bf16_bound_accepts_only_rounding_neighbors():
    # 1 + 1/256 is the midpoint between BF16 values 1 and 1 + 1/128.
    ref = torch.tensor([[1.00390625], [0.0]])
    assert tune.compare_outputs(ref, torch.tensor([1.0], dtype=torch.bfloat16)) == 0
    assert (
        tune.compare_outputs(ref, torch.tensor([1.0078125], dtype=torch.bfloat16)) == 0
    )
    assert (
        tune.compare_outputs(ref, torch.tensor([1.015625], dtype=torch.bfloat16)) == 1
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("where", ["output", "reference", "magnitude"])
def test_accumulation_bound_rejects_nonfinite_values(value, where):
    ref = torch.tensor([[0.0], [1.0]])
    out = torch.tensor([0.0])
    if where == "output":
        out[0] = value
    else:
        ref[0 if where == "reference" else 1, 0] = value
    assert tune.compare_outputs(ref, out) == 1


def test_one_outlier_cannot_round_down_to_zero():
    ref = torch.zeros((2, 20000))
    out = torch.zeros(20000)
    out[123] = 1.0
    assert tune.compare_outputs(ref, out, printLog=False) == 0.0001


@pytest.mark.parametrize("kid", [9000])
def test_tune_route_preserves_id_scales_and_output(monkeypatch, kid):
    data = tune.generate_data(256, 256, 128, kid, device="cpu")
    out = torch.empty((256, 256), dtype=torch.bfloat16)
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return args[2]

    monkeypatch.setattr(tune, "opus_gemm", record)
    result = tune.run_bench(
        data["x"],
        data["w"],
        out,
        data["x_scale"],
        data["w_scale"],
        kid,
    )
    assert result is out and len(calls) == 1
    (x, w, y), kwargs = calls[0]
    assert kwargs["kid"] == kid
    assert kwargs["layout"] == "bpreshuffle"
    assert kwargs["x_scale"] is data["x_scale"]
    assert kwargs["w_scale"] is data["w_scale"]
    assert x.shape == (256, 128) and x.data_ptr() == data["x"].data_ptr()
    assert w.data_ptr() == data["w"].data_ptr() and y.data_ptr() == out.data_ptr()


def test_existing_opus_gemm_routes_9000_through_bpreshuffle_backend(monkeypatch):
    data = tune.generate_data(256, 256, 128, 9000, device="cpu")
    calls = []

    def launch(x, w, x_scale, w_scale, out, **kwargs):
        calls.append((x, w, x_scale, w_scale, out, kwargs))
        return out

    monkeypatch.setattr(
        "aiter.ops.opus.dispatch._a8w8_family._launch_a8w8_blockscale_bpreshuffle_gemm",
        launch,
    )
    result = opus_api.opus_gemm(
        data["x"],
        data["w"],
        data["out"],
        kid=9000,
        layout="bpreshuffle",
        x_scale=data["x_scale"],
        w_scale=data["w_scale"],
    )
    assert result is data["out"] and len(calls) == 1
    _, _, x_scale, w_scale, out, kwargs = calls[0]
    assert x_scale is data["x_scale"] and w_scale is data["w_scale"]
    assert out is data["out"]
    assert kwargs["kid"] == 9000
    assert kwargs["instance"] is kernels_list[9000]


@pytest.mark.parametrize("scale_dtype", [torch.float8_e8m0fnu, torch.uint8])
@pytest.mark.parametrize("packed", [False, True], ids=["column-major", "quant-packed"])
def test_opus_entry_accepts_quant_scale_views_without_copy(
    monkeypatch, scale_dtype, packed
):
    data = tune.generate_data(256, 512, 384, device="cpu")
    logical = data["x_scale"].view(scale_dtype)
    # The HIP K128 quantizer writes g*M+m into a contiguous [M,K/128]
    # Tensor when transpose_scale=True. Reproduce its metadata on the CPU.
    supplied = logical.T.view(*logical.shape) if packed else logical
    before = supplied.view(torch.uint8).clone()
    calls = []

    def raw_launch(x, w, x_scale, w_scale, out, kid):
        calls.append(kid)
        assert x_scale.shape == logical.shape
        assert x_scale.stride() == (1, 256)
        assert x_scale.dtype == scale_dtype
        assert x_scale.data_ptr() == supplied.data_ptr()
        # Nonuniform scales across both axes catch an accidental transpose or
        # contiguous copy that would change the kernel's g*M+m interpretation.
        torch.testing.assert_close(
            x_scale.view(torch.uint8), logical.view(torch.uint8)
        )
        assert w_scale is data["w_scale"]
        assert x.data_ptr() == data["x"].data_ptr()
        assert w.data_ptr() == data["w"].data_ptr()
        assert out.data_ptr() == data["out"].data_ptr()

    monkeypatch.setattr(
        "aiter.ops.opus.gemm_op_a8w8._opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
        raw_launch,
    )
    result = opus_api.opus_gemm(
        data["x"],
        data["w"],
        data["out"],
        kid=9000,
        layout="bpreshuffle",
        x_scale=supplied,
        w_scale=data["w_scale"],
    )
    assert result is data["out"] and calls == [9000]
    torch.testing.assert_close(supplied.view(torch.uint8), before)


@pytest.mark.parametrize("kid", [9000])
def test_shape_csv_ignores_previous_backend_results(tuner, tmp_path, kid):
    source, output = tmp_path / "source.csv", tmp_path / "tuned.csv"
    pd.DataFrame([shape_row(libtype="ck", kernelId=17, splitK=3, us=4.2)]).to_csv(
        source, index=False
    )
    args = tuner.parser.parse_args(["-i", str(source), "-o", str(output), "--mp", "1"])
    tuner.pre_process(args)
    assert list(tuner.untunedf.columns) == tuner.keys
    keys = tuple(tuner.untunedf.iloc[0])
    kernel_name = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid].name
    result = tuner.result_to_df([((keys, kid, 0, kernel_name), 8.0, 0.0)])
    tuner.result_to_csv(result, str(output))
    assert list(pd.read_csv(output).columns) == [
        "gfx",
        "cu_num",
        "M",
        "N",
        "K",
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    saved = tuner.get_tuned_gemm_list(str(output))
    assert saved.iloc[0].kernelId == kid and saved.iloc[0].libtype == "opus"
    assert saved.iloc[0].kernelName == kernel_name
    tuner.pre_process(args)
    assert tuner.untunedf.empty
    args.all = True
    tuner.pre_process(args)
    assert len(tuner.untunedf) == 1


def test_tasks_use_exact_kids_and_preallocated_output(tuner, monkeypatch):
    rows = tuner._normalize_rows(pd.DataFrame([shape_row()]))[tuner.keys]
    captured = []
    compiled = []
    monkeypatch.setattr(
        tune.generic_tune,
        "mp_tuner",
        lambda tasks, groups, *args, **kwargs: captured.extend(tasks),
    )
    monkeypatch.setattr(
        tune, "_ensure_kids_compiled", lambda kids: compiled.append(set(kids))
    )
    args = tuner.parser.parse_args(["--mp", "1", "--libtype", "opus"])
    tuner.tune(rows, pd.DataFrame(), args)
    assert compiled == [{9000, 9010, 9011, 9012, 9020}]
    assert {task[0][1] for task in captured} == {9000, 9010, 9011, 9012, 9020}
    for task in captured:
        assert task[1] is tune.generate_data and task[3] is tune.run_bench
        assert task[6] is tune.run_torch
        assert task[8] == {"with_bounds": True}
        assert task[-1] == ("out",)  # mp_tuner poisons outside the timed call.


def test_all_libtype_groups_generic_and_gfx950_opus_candidates(tuner, monkeypatch):
    rows = tuner._normalize_rows(pd.DataFrame([shape_row()]))[tuner.keys]
    captured, groups, compiled = [], [], []
    monkeypatch.setattr(
        tune.generic_tune,
        "mp_tuner",
        lambda tasks, in_data, *args, **kwargs: (
            captured.extend(tasks),
            groups.extend(in_data),
        )[0],
    )
    monkeypatch.setattr(
        tune, "_ensure_kids_compiled", lambda kids: compiled.append(set(kids))
    )
    monkeypatch.setattr(
        tuner, "get_asm_kernels", lambda *a: {(64, 128, 1): ["test_asm"]}
    )

    args = tuner.parser.parse_args(["--mp", "1", "--libtype", "all"])
    tuner.tune(rows, pd.DataFrame(), args)

    libtypes = {task[0][4] for task in captured}
    assert libtypes == {"ck", "cktile", "asm", "opus"}
    assert {task[0][1] for task in captured if task[0][4] == "opus"} == {9000, 9010, 9011, 9012, 9020}
    assert compiled == [{9000, 9010, 9011, 9012, 9020}]
    assert groups == [(len(captured), ())]
    assert all(task[1] is tune.generate_data for task in captured)
    assert {task[2] for task in captured} == {(256, 256, 256, 0)}
    assert all(len(task[0]) == 6 and task[0][-1] is True for task in captured)
    assert {task[4][0] for task in captured if task[0][4] == "ck"} == {
        tune._CK_BENCH_KEYS
    }
    assert {task[4][0] for task in captured if task[0][4] == "cktile"} == {
        tune._CK_BENCH_KEYS,
        tune._CK_ROWMAJOR_BENCH_KEYS,
    }
    assert {task[4][0] for task in captured if task[0][4] == "opus"} == {
        tune._BENCH_KEYS
    }
    assert {task[4][0] for task in captured if task[0][4] == "asm"} == {
        tune._ASM_BENCH_KEYS
    }
    assert all(task[6] is tune.run_torch for task in captured)
    assert all(task[8] == {"with_bounds": True} for task in captured)
    assert all(task[12] is tune.compare_outputs for task in captured)
    assert all(task[-1] == ("out",) for task in captured)


@pytest.mark.parametrize("kid", [0, 11, 12])
def test_cktile_scale_layout_matches_wrapper_contract(tuner, monkeypatch, kid):
    # Cover four-wave (wrapper transposes), eight-wave column-major and
    # eight-wave AQRowMajor. Nonsquare/nonuniform scales expose wrong layouts.
    m, n, k = 256, 512, 384
    data = tune.generate_data(m, n, k, device="cpu")
    tasks = tuner.get_gemm_a8w8_blockscale_cktile_tune_task(
        ("gfx950", 256, m, n, k), False, 0, True, [1, 2, 3, 4], {}
    )
    task = next(task for task in tasks if task[0][1] == kid)
    kernel = tune.generic_tune.candidate_kernels_cktile_dict[kid]
    scales = data[task[4][0][2]]
    # Reconstruct the logical matrix using the pointer/stride interpretation
    # in gemm_a8w8_blockscale_cktile_common.cuh, not tensor logical indexing.
    if kernel.is_eight_warp and kernel.AQRowMajor:
        decoded = scales
        assert task[4][0][2] == "x_scale_fp32"
    else:
        decoded = scales.view(k // 128, m).T
        assert task[4][0][2] == "x_scale_t_fp32"
    torch.testing.assert_close(decoded, data["x_scale"].float(), rtol=0, atol=0)
    torch.testing.assert_close(
        data["w_scale_fp32"], data["w_scale"].float(), rtol=0, atol=0
    )
    expected = tune.run_torch(*(data[key] for key in tune._REF_KEYS))
    actual = tune.generic_tune.run_torch(
        data["x"], data["weight"], decoded, data["w_scale_fp32"], dtype=torch.float32
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # Exact replay must select the same input layout as the tuning task.
    tuner.untunedf = pd.DataFrame(
        [{**saved_row(), "N": n, "K": k, "libtype": "cktile", "kernelId": kid}]
    )
    monkeypatch.setattr(tune, "generate_data", lambda *a, **kw: data)
    calls = []

    def run(x, weight, sa, sb, out, actual_kid, split_k, preshuffle):
        assert sa is scales and sb is data["w_scale_fp32"]
        assert weight is data["weight_shuffle"] and x is data["x"]
        assert (actual_kid, split_k, preshuffle) == (kid, 0, True)
        calls.append(actual_kid)
        out.copy_(expected)
        return out

    monkeypatch.setattr(tune.generic_tune, "run_gemm_a8w8_blockscale_cktile", run)
    monkeypatch.setattr(
        "aiter.test_common.run_perftest", lambda fn, *a, **kw: (fn(*a), 3.0)
    )
    result = tuner.run_config(tuner.parser.parse_args(["--mp", "1"]))
    assert calls == [kid] and result[0]["status"] == "ok"


@pytest.mark.parametrize("libtype", ["ck", "cktile", "both", "asm", "all"])
def test_common_backends_cover_shapes_without_opus_candidates(
    tuner, monkeypatch, libtype
):
    # M=32 is valid for common backends but below every OPUS tile's M alignment.
    rows = tuner._normalize_rows(pd.DataFrame([shape_row(M=32, N=128)]))[tuner.keys]
    captured = []
    monkeypatch.setattr(
        tuner, "get_asm_kernels", lambda *a: {(64, 128, 1): ["test_asm"]}
    )
    monkeypatch.setattr(
        tune.generic_tune,
        "mp_tuner",
        lambda tasks, *args, **kwargs: captured.extend(tasks),
    )
    with patch.object(tune, "_ensure_kids_compiled") as compile_opus:
        args = tuner.parser.parse_args(["--mp", "1", "--libtype", libtype, "--splitK"])
        tuner.tune(rows, pd.DataFrame(), args)
    compile_opus.assert_not_called()
    expected = {"ck", "cktile"} if libtype == "both" else {libtype}
    if libtype == "all":
        expected = {"ck", "cktile", "asm"}
    assert {task[0][4] for task in captured} == expected
    for task in captured:
        if task[0][4] == "asm":
            assert task[0][2] in tune.generic_tune.get_valid_asm_splitK_list(256, 8)
        else:
            assert task[0][2] == 0  # CK/CKTile B-preshuffle does not tune split-K.


def test_mixed_backend_winners_and_profile_round_trip(tuner, tmp_path):
    results = []
    backends = ("ck", "cktile", "asm", "opus")
    for index, winner in enumerate(backends):
        keys = ("gfx950", 256, 256 * (index + 1), 256, 256)
        for backend in backends:
            kid = 9000 if backend == "opus" else 0
            name = "test_asm" if backend == "asm" else tuner.getKernelName(kid, backend)
            info = (keys, kid, int(backend == "asm"), name, backend, True)
            results.append((info, 2.0 if backend == winner else 4.0, 0.0))
        # A faster candidate with a numerical mismatch must never win.
        results.append((results[-1][0], 1.0, 0.01))
    output, profile = tmp_path / "mixed.csv", tmp_path / "profile.csv"
    args = tuner.parser.parse_args(["-o2", str(profile), "--mp", "1"])
    winners = tuner.post_process(results, args, topk=1)
    tuner.result_to_csv(winners, str(output))
    saved = tuner.get_tuned_gemm_list(str(output))
    assert saved.libtype.tolist() == list(backends)
    assert saved.us.tolist() == [2.0] * 4
    assert saved.errRatio.eq(0).all()
    assert list(saved.columns) == tuner.columns
    assert len(pd.read_csv(profile)) == len(results)


@pytest.mark.parametrize(
    "backends", [("ck", "cktile", "asm", "opus"), ("ck", "cktile")]
)
def test_mixed_exact_replay_preserves_backends_and_avoids_production_config(
    tuner, monkeypatch, tmp_path, backends
):
    data = tune.generate_data(256, 256, 256, device="cpu")
    ref = tune.run_torch(*(data[key] for key in tune._REF_KEYS))
    rows = []
    calls = []
    for backend in backends:
        kid = 9000 if backend == "opus" else 0
        rows.append(
            {
                **saved_row(),
                "libtype": backend,
                "kernelId": kid,
                "splitK": int(backend == "asm"),
                "us": 3.0,
                "tflops": 0.0,
                "bw": 0.0,
                "errRatio": 0.0,
                "kernelName": (
                    "test_asm"
                    if backend == "asm"
                    else tuner.getKernelName(kid, backend)
                ),
            }
        )
    # A mixed-device CSV must replay only the current device's rows.
    rows.append({**rows[0], "gfx": "gfx942", "cu_num": 304})
    source = tmp_path / "mixed.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    monkeypatch.setattr(tune, "generate_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        tuner, "get_asm_kernels", lambda *a: {(64, 128, 1): ["test_asm"]}
    )

    def bench(backend):
        def run(*args):
            keys = (
                tune._BENCH_KEYS
                if backend == "opus"
                else tune._ASM_BENCH_KEYS if backend == "asm" else tune._CK_BENCH_KEYS
            )
            assert all(actual is data[key] for actual, key in zip(args, keys))
            expected = (
                (9000,)
                if backend == "opus"
                else (("test_asm", 1, True) if backend == "asm" else (0, 0, True))
            )
            assert args[len(keys) :] == expected
            assert torch.isnan(data["out"]).all()
            data["out"].copy_(ref)
            calls.append(backend)
            return data["out"]

        return run

    monkeypatch.setattr(tune, "run_bench", bench("opus"))
    for backend, function in (
        ("ck", "run_gemm_a8w8_blockscale"),
        ("cktile", "run_gemm_a8w8_blockscale_cktile"),
        ("asm", "run_gemm_a8w8_blockscale_asm"),
    ):
        monkeypatch.setattr(tune.generic_tune, function, bench(backend))
    monkeypatch.setattr(
        "aiter.test_common.run_perftest", lambda fn, *args, **kwargs: (fn(*args), 3.0)
    )
    with (
        patch.object(tune, "_ensure_kids_compiled") as compile_opus,
        patch.object(tuner, "_set_config_env_for_run_config") as production_config,
    ):
        tuner.run(tuner.parser.parse_args(["--run_config", str(source), "--mp", "1"]))
    assert calls == list(backends)
    if "opus" in backends:
        compile_opus.assert_called_once_with({9000})
    else:
        compile_opus.assert_not_called()
    production_config.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"libtype": "unknown"},
        {"libtype": "ck", "kernelId": 0, "splitK": 1},
        {"libtype": "cktile", "kernelId": 0, "splitK": 1},
        {"libtype": "cktile", "kernelId": 99999},
        {"libtype": "ck", "kernelId": 0, "kernelName": "stale_name"},
        {"libtype": "opus", "kernelName": "stale_name"},
        {"libtype": "asm", "kernelId": 0, "splitK": 1},
        {"libtype": "asm", "kernelId": 0, "splitK": 1, "kernelName": "unknown"},
        {"libtype": "asm", "kernelId": 0, "splitK": 0, "kernelName": "test_asm"},
        {"libtype": "asm", "kernelId": 0, "splitK": 3, "kernelName": "test_asm"},
        {"gfx": "gfx942"},
    ],
)
def test_invalid_mixed_replay_rejects_before_build_or_allocation(
    tuner, monkeypatch, changes
):
    tuner.untunedf = pd.DataFrame([{**saved_row(), **changes}])
    monkeypatch.setattr(
        tuner, "get_asm_kernels", lambda *a: {(64, 128, 1): ["test_asm"]}
    )
    with (
        patch.object(tune, "generate_data") as generate,
        patch.object(tune, "_ensure_kids_compiled") as compile_opus,
        pytest.raises(ValueError),
    ):
        tuner.run_config(tuner.parser.parse_args(["--mp", "1"]))
    generate.assert_not_called()
    compile_opus.assert_not_called()


@pytest.mark.parametrize("shape", [(256, 127, 128), (256, 128, 129)])
def test_invalid_common_layout_rejects_before_allocation(tuner, shape):
    with pytest.raises(ValueError, match="divisible"):
        tuner._normalize_rows(
            pd.DataFrame([shape_row(M=shape[0], N=shape[1], K=shape[2])])
        )
    with (
        patch.object(torch, "randn") as allocate,
        pytest.raises(ValueError, match="divisible"),
    ):
        tune.generate_data(*shape, device="cpu")
    allocate.assert_not_called()


def test_reference_handles_partial_n_scale_block():
    x = torch.ones((2, 128)).to(torch.float8_e4m3fn)
    w = torch.ones((144, 128)).to(x.dtype)
    sa = torch.tensor([[1.0], [2.0]]).to(torch.float8_e8m0fnu)
    sb = torch.tensor([[0.5], [4.0]]).to(torch.float8_e8m0fnu)
    ref = tune.run_torch(x, w, sa, sb)
    expected = torch.tensor([1.0, 2.0])[:, None] * torch.tensor(
        [64.0] * 128 + [512.0] * 16
    )
    torch.testing.assert_close(ref, expected)


def test_generic_libtype_options_are_available_and_preshuffle_is_fixed(tuner):
    args = tuner.parser.parse_args(["--mp", "1"])
    assert args.libtype == "all"
    assert args.preshuffle is True
    for libtype in ("ck", "cktile", "asm", "opus", "both", "all"):
        assert (
            tuner.parser.parse_args(["--mp", "1", "--libtype", libtype]).libtype
            == libtype
        )


@pytest.mark.parametrize("kid", [9000])
def test_replay_calls_saved_exact_kid_without_conversion(tuner, monkeypatch, kid):
    data = tune.generate_data(256, 256, 256, kid, device="cpu")
    ref = tune.run_torch(*(data[k] for k in tune._REF_KEYS))
    row = {**saved_row(), "kernelId": kid}
    for column in ("dtype", "outdtype", "scale_dtype"):
        row.pop(column)
    tuner.untunedf = pd.DataFrame([row])
    calls = []
    compiled = []

    def exact(x, w, out, x_scale, w_scale, actual_kid):
        assert actual_kid == kid
        assert x_scale is data["x_scale"] and w_scale is data["w_scale"]
        assert torch.isnan(out).all()
        out.copy_(ref)
        calls.append(actual_kid)
        return out

    monkeypatch.setattr(tune, "generate_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(tune, "run_bench", exact)
    monkeypatch.setattr(
        tune, "_ensure_kids_compiled", lambda kids: compiled.append(set(kids))
    )
    monkeypatch.setattr(
        "aiter.test_common.run_perftest", lambda fn, *args, **kwargs: (fn(*args), 3.0)
    )
    results = tuner.run_config(tuner.parser.parse_args(["--mp", "1"]))
    assert compiled == [{9000}]
    assert len(calls) == 1 and results[0]["status"] == "ok"


@pytest.mark.parametrize(
    "change",
    [
        {"scale_dtype": "fp32"},
        {"libtype": "ck"},
        {"splitK": 2},
        {"outdtype": "fp16"},
        {"kernelId": 9000.5},
        {"kernelId": 1},
        {"kernelId": 11000},
    ],
)
def test_stale_or_incompatible_saved_rows_reject_before_data_generation(tuner, change):
    tuner.untunedf = pd.DataFrame([{**saved_row(), **change}])
    with patch.object(tune, "generate_data") as generate, pytest.raises(ValueError):
        tuner.run_config(tuner.parser.parse_args(["--mp", "1"]))
    generate.assert_not_called()


def test_wrong_arch_rejects_before_reading_csv(tuner, monkeypatch):
    monkeypatch.setattr(tuner, "get_gfx", lambda: "gfx942")
    with (
        patch.object(tuner, "get_untuned_gemm_list") as read,
        pytest.raises(SystemExit),
    ):
        tuner.pre_process(tuner.parser.parse_args(["-i", "missing.csv", "--mp", "1"]))
    read.assert_not_called()


@pytest.mark.parametrize("kid", [9000, 9010, 9011, 9012])
@pytest.mark.parametrize(
    "shape",
    [
        (256, 256, 128),   # Prologue and final tile only.
        (1280, 512, 384),  # Rectangular grid without the 2x2 tile swizzle.
        (512, 1024, 256),  # Rectangular grid with the 2x2 tile swizzle.
        (256, 512, 8320),  # Refill after the first 64 K128 scale groups.
    ],
)
def test_gpu_kernel_matches_independent_reference(kid, shape):
    if not torch.cuda.is_available():
        pytest.skip("gfx950 GPU required")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 GPU required")
    data = tune.generate_data(*shape, kid, device="cuda:0")
    if shape[2] > 8192:
        # Isolate the refilled scale panel for a strict short-accumulation
        # check. Full random long-K inputs are checked against the existing
        # source accumulation contract below, including cancellation.
        data["x"][:, :8192] = 0
    ref = tune.run_torch(*(data[key] for key in tune._REF_KEYS))
    data["out"] = torch.full(
        shape[:2], float("nan"), device="cuda:0", dtype=torch.bfloat16
    )
    tune.run_bench(*(data[key] for key in tune._BENCH_KEYS), kid)
    torch.testing.assert_close(data["out"].float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("kid", [9000, 9010, 9011, 9012])
@pytest.mark.parametrize("shape", [(2048, 768, 7168), (256, 512, 8320)])
def test_gpu_large_shape_meets_accumulation_bounds(kid, shape):
    if not torch.cuda.is_available():
        pytest.skip("gfx950 GPU required")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 GPU required")
    data = tune.generate_data(*shape, kid, device="cuda:0")
    ref = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
    data["out"] = torch.full(
        shape[:2], float("nan"), device="cuda:0", dtype=torch.bfloat16
    )
    tune.run_bench(*(data[key] for key in tune._BENCH_KEYS), kid)
    assert tune.compare_outputs(ref, data["out"]) == 0


@pytest.mark.parametrize(
    "kid,shape",
    [
        (9010, (128, 128, 128)),
        (9010, (128, 256, 384)),
        (9011, (64, 128, 256)),
        (9011, (192, 384, 8320)),
        # Unified schedule: 31/32 K groups and 256/272 CTA grids.
        (9011, (192, 384, 3968)),
        (9011, (192, 384, 4096)),
        (9011, (1024, 2048, 7168)),
        (9011, (1088, 2048, 7168)),
        # Fixed N/K isolates the 240/256/272 CTA mapping boundary;
        # all three use the interleaved schedule at K=128.
        (9011, (960, 2048, 128)),
        (9011, (1024, 2048, 128)),
        (9011, (1088, 2048, 128)),
        # M/N mapping thresholds, including small grids that stay linear.
        (9011, (1472, 2048, 128)),
        (9011, (1536, 2048, 256)),
        (9011, (1600, 2048, 384)),
        (9011, (1536, 1024, 128)),
        (9011, (1536, 1152, 128)),
        (9011, (1536, 2176, 128)),
        (9011, (1088, 768, 128)),
        (9011, (1600, 1152, 128)),  # 225 CTAs: small-grid linear path.
        # Nondivisible mapping fallbacks must exceed the 256-CTA cutoff.
        (9011, (1984, 1152, 128)),  # 31x9 = 279 CTAs.
        (9011, (2816, 768, 128)),   # 44x6 = 264 CTAs.
        # Narrow-N 752/768 CTA boundary; either partition policy must
        # preserve the same numerical contract.
        (9011, (6016, 1024, 128)),
        (9011, (6144, 1024, 128)),
        # The single four-stage 9012 pipeline covers short/long K and grids
        # on both sides of the former CU boundaries, with independent scales.
        (9012, (192, 384, 3968)),
        (9012, (192, 384, 4096)),
        # Four-stage prefetch: every remainder after the unrolled rotation.
        (9012, (192, 384, 4224)),
        (9012, (192, 384, 4352)),
        (9012, (192, 384, 4480)),
        (9012, (512, 2048, 7168)),
        (9012, (576, 2048, 7168)),
        # Seventh CK target uses the same four-stage kernel as the first six.
        (9012, (1408, 768, 7168)),
        (9012, (1024, 2048, 7168)),
        (9012, (1088, 2048, 7168)),
        (9012, (64, 128, 128)),
        (9012, (64, 128, 256)),   # Two K tiles, no LDS-slot reuse.
        (9012, (192, 384, 768)),  # Four LDS slots, one A/B register buffer.
        (9012, (192, 384, 8320)),
        (9012, (192, 384, 16512)),  # Third scale panel and an odd final drain.
    ] + [
        # The 9010 register-replacement loop has a separate final tile and
        # refills its scale panel before consuming the next K128 group.
        (9010, (128, 384, k))
        for k in (256, 512, 640, 768, 896, 1024, 8064, 8192, 8320, 8448, 16512)
    ] + [
        (kid, (192, 384, k))
        for kid in (9011, 9012)
        # Exercise the steady-loop entry, each drain position, and both
        # sides of the scale-panel boundary without changing tolerances.
        for k in (384, 512, 640, 896, 1024, 8064, 8192, 8448, 8576)
    ],
)
def test_gpu_small_tiles_cover_minimum_grids_and_scale_groups(kid, shape):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 GPU required")
    data = tune.generate_data(*shape, kid, device="cuda:0")
    ref = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
    data["out"].fill_(float("nan"))
    tune.run_bench(*(data[key] for key in tune._BENCH_KEYS), kid)
    assert tune.compare_outputs(ref, data["out"]) == 0


@pytest.mark.parametrize(
    "m,k", [(64, 128), (128, 256), (192, 384), (1088, 7168), (1152, 384),
            (1216, 8320), (1984, 8320), (1024, 256)],
)
def test_gpu_padded_m_matches_reference_and_preserves_output_canary(m, k):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 GPU required")
    n = 512
    data = tune.generate_data(m, n, k, device="cuda:0")
    ref = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
    padded_m = (m + 255) // 256 * 256
    storage = torch.full((padded_m + 64, n), float("nan"), device="cuda:0", dtype=torch.bfloat16)
    data["out"] = storage[:m]
    tune.run_bench(*(data[key] for key in tune._BENCH_KEYS), 9020)
    assert tune.compare_outputs(ref, data["out"]) == 0
    assert torch.isnan(storage[m:]).all(), "The padded M rows must not be stored"
