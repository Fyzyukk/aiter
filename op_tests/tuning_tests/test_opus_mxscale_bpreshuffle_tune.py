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
    assert kids == [9000]
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
    root = Path(__file__).resolve().parents[2]
    command = [
        sys.executable,
        str(root / "csrc/opus_gemm/gen_instances.py"),
        "--working_path",
        str(tmp_path),
    ]
    if explicit_request:
        command += ["--extra_kids", "9000"]
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
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_BF16_SIZE 1"
        in dispatch
    )
    assert (
        "#define "
        "GENERATE_A8W8_BLOCKSCALE_BPRESHUFFLE_KID_DISPATCH_GFX950_FP32_SIZE 0"
        in dispatch
    )
    for kid in (9000,):
        name = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid].name
        assert name in manifest
        assert f"{{ {kid}, &{name}<bf16_t> }}" in dispatch
        assert f"{name}<bf16_t>" in host
        assert f"{{ {kid}, &{name}<fp32_t> }}" not in dispatch
        assert not (tmp_path / f"instances/{name}_Cfp32_t.device.cu").exists()
    four_wave_name = a8w8_mxscale_gemm_bpreshuffle_kernels_list[9000].name
    four_wave = (tmp_path / f"impl/{four_wave_name}.cuh").read_text()
    assert "gemm_a8w8_mxfp8_scale_kernel" in four_wave
    assert "blockscale_generic" not in four_wave
    assert "if (args.k == " not in four_wave
    assert "blockscale_panel::gemm_a8w8_mxfp8_scale_kernel" not in four_wave
    assert f"{four_wave_name}_WholeKTraits" not in four_wave
    device = (tmp_path / f"instances/{four_wave_name}_Cbf16_t.device.cu").read_text()
    assert f"{four_wave_name}_Traits<bf16_t>" in device
    assert "gemm_a8w8_mxfp8_scale_kernel" in device
    assert "blockscale_generic" not in device
    assert "blockscale_panel::gemm_a8w8_mxfp8_scale_kernel" not in device


def test_tuner_reuses_existing_jit_module_and_public_entry():
    root = Path(__file__).resolve().parents[2]
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


def test_native_e8m0_data_and_standard_preshuffle():
    data = tune.generate_data(256, 512, 256, 9000, device="cpu")
    sa, sb = data["x_scale"], data["w_scale"]
    assert data["x"].dtype == data["w"].dtype == torch.float8_e4m3fn
    assert sa.dtype == sb.dtype == torch.float8_e8m0fnu
    assert sa.shape == (256, 2) and sa.stride() == (1, 256)
    assert sb.shape == (4, 2) and sb.is_contiguous()
    assert torch.unique(sa.view(torch.uint8)).numel() > 1
    assert torch.unique(sb.view(torch.uint8)).numel() > 1
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
        tune, "mp_tuner", lambda tasks, groups, **kwargs: captured.extend(tasks)
    )
    monkeypatch.setattr(
        tune, "_ensure_kids_compiled", lambda kids: compiled.append(set(kids))
    )
    args = tuner.parser.parse_args(["--mp", "1"])
    tuner.tune(rows, pd.DataFrame(), args)
    assert compiled == [{9000}]
    assert {task[0][1] for task in captured} == {9000}
    for task in captured:
        assert task[1] is tune.generate_data and task[3] is tune.run_bench
        assert task[6] is tune.run_torch
        assert task[8] == {"with_bounds": True}
        assert task[-1] == ("out",)  # mp_tuner poisons outside the timed call.


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


@pytest.mark.parametrize("kid", [9000])
@pytest.mark.parametrize("shape", [(256, 256, 128), (1280, 512, 384)])
def test_gpu_kernel_matches_independent_reference(kid, shape):
    if not torch.cuda.is_available():
        pytest.skip("gfx950 GPU required")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 GPU required")
    data = tune.generate_data(*shape, kid, device="cuda:0")
    ref = tune.run_torch(*(data[key] for key in tune._REF_KEYS))
    data["out"] = torch.full(
        shape[:2], float("nan"), device="cuda:0", dtype=torch.bfloat16
    )
    tune.run_bench(*(data[key] for key in tune._BENCH_KEYS), kid)
    torch.testing.assert_close(data["out"].float(), ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("kid", [9000])
def test_gpu_large_shape_meets_accumulation_bounds(kid):
    if not torch.cuda.is_available():
        pytest.skip("gfx950 GPU required")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("gfx950 GPU required")
    data = tune.generate_data(2048, 768, 7168, kid, device="cuda:0")
    ref = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
    data["out"] = torch.full(
        (2048, 768), float("nan"), device="cuda:0", dtype=torch.bfloat16
    )
    tune.run_bench(*(data[key] for key in tune._BENCH_KEYS), kid)
    assert tune.compare_outputs(ref, data["out"]) == 0
