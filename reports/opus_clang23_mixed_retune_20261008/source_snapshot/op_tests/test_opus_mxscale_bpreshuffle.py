import pytest
import torch

from csrc.opus_gemm.opus_gemm_common import (
    A8W8_BPRESHUFFLE_TUNING_KIDS,
    a8w8_mxscale_bpreshuffle_supports_shape,
    a8w8_mxscale_gemm_bpreshuffle_kernels_list,
)


def test_grouped_bpreshuffle_contract():
    kernels = a8w8_mxscale_gemm_bpreshuffle_kernels_list
    instance = kernels[9020]
    assert instance.name.endswith("_main")
    assert len({kernel.name for kernel in kernels.values()}) == len(kernels)
    for shape in (
        (352, 65536, 1536),
        (384, 65536, 1536),
        (512, 2304, 16384),
        (6144, 768, 128),
    ):
        assert a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)
    for shape in ((353, 256, 128), (16, 128, 128), (16, 256, 16512)):
        assert not a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)


@pytest.mark.parametrize(
    "shape",
    [
        (16, 256, 128),
        (176, 65792, 128),
        (208, 65792, 384),
        (352, 65536, 1536),
        (384, 65536, 1536),
        (16, 256, 4224),
        (16, 256, 8320),
        (512, 2304, 16384),
        (6144, 768, 128),
    ],
)
def test_grouped_bpreshuffle_signed_tails(shape):
    _check_signed_repeated(9020, shape, 3)


def test_small_bpreshuffle_registry():
    from csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune import (
        candidate_kids_for_shape,
    )

    kernels = a8w8_mxscale_gemm_bpreshuffle_kernels_list
    assert len({instance.name for instance in kernels.values()}) == len(kernels)
    for kid in range(9040, 9057):
        instance = kernels[kid]
        assert instance.direct_only and instance.scale_dtype == "e8m0"
        max_m = 2048 if kid in {9043, 9044, 9045, 9046, 9055, 9056} else 512
        assert instance.max_m == max_m and instance.max_k == 16384
        assert (kid in candidate_kids_for_shape("gfx950", 511, 256, 1152)) == (
            kid in A8W8_BPRESHUFFLE_TUNING_KIDS
        )
        assert kid in candidate_kids_for_shape(
            "gfx950", max_m, 256, 128, include_legacy=True,
        )
        for shape in ((max_m + 1, 256, 128), (16, 128, 16512), (16, 128, 129)):
            assert not a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)


@pytest.mark.parametrize("kid", [9041, 9042, 9043, 9044, 9046, 9047, 9048, 9049])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 128, 128),
        (17, 128, 384),
        (33, 256, 640),
        (96, 7168, 768),
        (128, 7168, 1024),
        (32, 65536, 1536),
        (512, 7168, 896),
        (511, 256, 1152),
        (512, 256, 16384),
    ],
)
def test_small_bpreshuffle_signed_ring(kid, shape):
    _check_signed_repeated(kid, shape, 16)


@pytest.mark.parametrize("kid", [9040, 9042, 9043, 9045, 9046])
@pytest.mark.parametrize("shape", [(97, 768, 1152), (97, 768, 16384)])
def test_small_bpreshuffle_shared_paths(kid, shape):
    _check_signed_repeated(kid, shape, 16)


@pytest.mark.parametrize("kid", range(9050, 9057))
@pytest.mark.parametrize(
    "shape", [(1, 128, 128), (33, 256, 640), (511, 256, 1152), (97, 768, 16384)]
)
def test_optimized_bpreshuffle_signed_prefetch(kid, shape):
    _check_signed_repeated(kid, shape, 16)


@pytest.mark.parametrize("kid", [9043, 9044, 9045, 9046, 9055, 9056])
@pytest.mark.parametrize(
    "shape",
    [(513, 128, 128), (1023, 256, 640), (1729, 768, 7168),
     (2047, 128, 1152), (2048, 256, 16384)],
)
def test_extended_bpreshuffle_signed_tails(kid, shape):
    _check_signed_repeated(kid, shape, 16)


@pytest.mark.parametrize("kid", [9043, 9044, 9045, 9046, *range(9050, 9057)])
def test_optimized_bpreshuffle_cancellation(kid):
    m = 513 if a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid].max_m > 512 else 97
    _check_signed_repeated(kid, (m, 256, 1280), 16, cancellation=True)


@pytest.mark.parametrize("backend", ["ck", "cktile", "asm"])
def test_tuner_preserves_external_accuracy_contract(backend, monkeypatch):
    from pathlib import Path

    from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune

    generic = tune.generic_tune
    monkeypatch.setattr(generic, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(
        generic, "get_asm_dir",
        lambda: str(Path(__file__).resolve().parents[1] / "hsa" / "gfx950"),
    )
    tuner = tune.OpusMxscaleBpreshuffleTuner()
    name = {
        "ck": "get_gemm_a8w8_blockscale_tune_task",
        "cktile": "get_gemm_a8w8_blockscale_cktile_tune_task",
        "asm": "get_gemm_a8w8_blockscale_asm_tune_task",
    }[backend]
    params = [("gfx950", 256, 128, 768, 7168), True, 0, True]
    if backend == "cktile":
        params.append([1, 2, 3, 4])
    params.append({"num_warmup": 1, "num_iters": 1})
    original = getattr(generic.GemmA8W8BlockScaleTuner, name)(tuner, *params)
    adapted = getattr(tuner, name)(*params)
    assert len(adapted) == len(original) > 0
    for old, new in zip(original, adapted):
        assert new[:4] == old[:4]
        assert new[5:] == old[5:]
        assert new[6] is generic.run_torch
        assert new[12] is None  # The original checkAllclose path.
        if backend != "cktile":
            assert new[4] == old[4]


def test_tuner_selects_with_backend_accuracy_limits_and_keeps_raw_errors(tmp_path):
    import argparse
    import csv

    from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune

    tuner = tune.OpusMxscaleBpreshuffleTuner()
    assert tuner.parser.get_default("errRatio") == 0.05
    shape = ("gfx950", 256, 128, 768, 7168)
    rows = [
        ((shape, 9000, 0, "", "opus", True), 1.0, 0.0001),
        ((shape, 9021, 0, "", "opus", True), 3.0, 0.0),
        ((shape, 0, 4, "asm_candidate", "asm", True), 2.0, 0.003),
        ((shape, 1, 4, "asm_rejected", "asm", True), 0.5, 0.06),
    ]
    profile = tmp_path / "profile.csv"
    args = argparse.Namespace(profile_file=str(profile), errRatio=0.05, verbose=False)
    selected = tuner.post_process(rows, args, topk=1)
    assert selected.iloc[0]["libtype"] == "asm"
    assert selected.iloc[0]["us"] == 2.0
    assert selected.iloc[0]["errRatio"] == 0.003
    with profile.open(newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == len(rows)
    rejected_opus = next(row for row in saved if row["kernelId"] == "9000")
    assert float(rejected_opus["us"]) == 1.0
    assert float(rejected_opus["errRatio"]) == 0.0001


def test_external_only_tuning_retains_original_llvm(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pandas as pd
    from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune

    # An external-only run neither needs nor selects the OPUS toolchain.
    monkeypatch.setenv("OPUS_HIP_CLANG_PATH", str(tmp_path / "missing-opus-llvm"))
    monkeypatch.setenv("HIP_CLANG_PATH", "/previous/compiler")
    monkeypatch.delenv("AITER_HIP_RESOURCE_DIR", raising=False)
    monkeypatch.setattr(tune.subprocess, "check_output", lambda *a, **kw: pytest.fail("OPUS compiler probed"))
    tuner = tune.OpusMxscaleBpreshuffleTuner()
    monkeypatch.setattr(tuner, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(tuner, "get_cu_num", lambda: 256)
    monkeypatch.setattr(tuner, "get_gemm_a8w8_blockscale_cktile_tune_task", lambda *a: ["task"])
    tasks = []
    monkeypatch.setattr(tune.generic_tune, "mp_tuner", lambda work, *a, **kw: tasks.extend(work) or [])
    args = SimpleNamespace(opus_kids=None, libtype="cktile", warmup=5, iters=51,
                           splitK=True, blockPerCu=[1], mp=1, shape_grouped=True,
                           errRatio=.05, timeout=None, verbose=False)
    tuner.tune(pd.DataFrame([{"M": 4096, "N": 2048, "K": 7168}]), None, args)
    assert tasks == ["task"]
    assert tune.os.environ["HIP_CLANG_PATH"] == "/previous/compiler"
    assert "AITER_HIP_RESOURCE_DIR" not in tune.os.environ


@pytest.mark.parametrize("previous_compiler", [None, "/external/compiler"])
@pytest.mark.parametrize("build_fails", [False, True])
def test_opus_build_restores_external_compiler(monkeypatch, tmp_path, previous_compiler, build_fails):
    from types import SimpleNamespace

    from aiter.jit import core
    from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune

    compiler_dir = tmp_path / "pin-llvm" / "bin"
    compiler_dir.mkdir(parents=True)
    (compiler_dir / "clang++").touch()
    resource_dir = tmp_path / "opus-resources"
    (resource_dir / "include").mkdir(parents=True)
    monkeypatch.setenv("OPUS_HIP_CLANG_PATH", str(compiler_dir))
    monkeypatch.setenv("OPUS_HIP_RESOURCE_DIR", str(resource_dir))
    monkeypatch.setenv("AITER_HIP_RESOURCE_DIR", "/external/resources")
    if previous_compiler is None:
        monkeypatch.delenv("HIP_CLANG_PATH", raising=False)
    else:
        monkeypatch.setenv("HIP_CLANG_PATH", previous_compiler)
    monkeypatch.setattr(tune.subprocess, "check_output", lambda *a, **kw: "clang version 24.0.0")
    cleared = []
    for name in ("hip_flag_checker", "check_LLVM_MAIN_REVISION"):
        monkeypatch.setattr(core, name, SimpleNamespace(cache_clear=lambda n=name: cleared.append(n)))

    def build(kids):
        assert kids == frozenset({9000, 9001, 9010, 9011})
        assert tune.os.environ["HIP_CLANG_PATH"] == str(compiler_dir)
        assert tune.os.environ["AITER_HIP_RESOURCE_DIR"] == str(resource_dir)
        if build_fails:
            raise RuntimeError("diagnostic build failure")
        return True

    monkeypatch.setitem(tune.sys.modules, "opus_gemm_tune", SimpleNamespace(_ensure_kids_compiled=build))
    if build_fails:
        with pytest.raises(RuntimeError, match="diagnostic build failure"):
            tune._ensure_kids_compiled({9000, 9001, 9010, 9011})
    else:
        assert tune._ensure_kids_compiled({9000, 9001, 9010, 9011}) is True
    assert tune.os.environ.get("HIP_CLANG_PATH") == previous_compiler
    assert tune.os.environ["AITER_HIP_RESOURCE_DIR"] == "/external/resources"
    assert cleared == ["hip_flag_checker", "check_LLVM_MAIN_REVISION"] * 2


def test_tuning_rejects_missing_compiler_without_system_fallback(monkeypatch, tmp_path):
    from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune

    monkeypatch.setenv("OPUS_HIP_CLANG_PATH", str(tmp_path / "missing"))
    monkeypatch.setenv("HIP_CLANG_PATH", "/previous/compiler")
    with pytest.raises(FileNotFoundError, match="OPUS_HIP_CLANG_PATH"):
        with tune._opus_compiler_environment():
            pytest.fail("Missing OPUS compiler accepted")
    assert tune.os.environ["HIP_CLANG_PATH"] == "/previous/compiler"


def test_opus_resource_headers_are_scoped_to_build(monkeypatch, tmp_path):
    import cpp_extension
    from aiter.jit import core
    from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune

    compiler_dir = tmp_path / "pin-llvm" / "bin"
    compiler_dir.mkdir(parents=True)
    (compiler_dir / "clang++").touch()
    resource_dir = tmp_path / "rocm" / "lib" / "llvm" / "lib" / "clang" / "20"
    (resource_dir / "include").mkdir(parents=True)
    monkeypatch.setenv("OPUS_HIP_CLANG_PATH", str(compiler_dir))
    for key in ("HIP_CLANG_PATH", "OPUS_HIP_RESOURCE_DIR", "AITER_HIP_RESOURCE_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tune.subprocess, "check_output", lambda *a, **kw: "clang version 24.0.0")
    monkeypatch.setattr(core, "get_hip_version", lambda: "7.0.0")
    monkeypatch.setattr(cpp_extension, "ROCM_HOME", str(tmp_path / "rocm"))
    with tune._opus_compiler_environment():
        assert tune.os.environ["AITER_HIP_RESOURCE_DIR"] == str(resource_dir)
        assert tune.os.environ["HIP_CLANG_PATH"] == str(compiler_dir)
    assert "AITER_HIP_RESOURCE_DIR" not in tune.os.environ
    assert "HIP_CLANG_PATH" not in tune.os.environ


def _check_signed_repeated(kid, shape, repetitions, *, cancellation=False):
    if not torch.cuda.is_available():
        pytest.skip("requires gfx950")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")

    from aiter.ops.shuffle import shuffle_weight
    from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune

    tune._ensure_kids_compiled({kid})
    m, n, k = shape
    data = tune.generate_data(m, n, k, 17, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(29)
    for name in ("x", "w_reference"):
        value = data[name]
        signs = torch.randint(0, 2, value.shape, device="cuda", generator=generator)
        data[name] = (value.float() * (2 * signs - 1)).to(value.dtype)
    if cancellation:
        # Matching products with opposite signs in the two K halves exercise
        # both scale indexing and FP32 reduction before BF16 conversion.
        assert k % 256 == 0
        half = k // 2
        for name in ("x", "w_reference"):
            bits = data[name].view(torch.uint8)
            bits[:, half:] = bits[:, :half] ^ (128 if name == "w_reference" else 0)
        for name in ("x_scale", "w_scale"):
            bits = data[name].view(torch.uint8)
            bits[:, k // 256:] = bits[:, :k // 256]
    data["w"] = shuffle_weight(data["w_reference"], layout=(16, 16))
    reference = tune.run_torch(
        *(data[name] for name in tune._REF_KEYS), with_bounds=True
    )
    storage = torch.full((m * n + 256,), 42, device="cuda", dtype=torch.bfloat16)
    data["out"] = storage[128:-128].view(m, n)
    first = None
    for _ in range(repetitions):
        data["out"].fill_(float("nan"))
        result = tune.run_bench(*(data[name] for name in tune._BENCH_KEYS), kid)
        assert tune.compare_outputs(reference, result, printLog=False) == 0
        assert torch.all(storage[:128] == 42)
        assert torch.all(storage[-128:] == 42)
        if first is None:
            first = result.clone()
        else:
            assert torch.equal(first, result)


@pytest.mark.parametrize("kid", range(9060, 9070))
@pytest.mark.parametrize("shape", [
    (1, 128, 128), (81, 256, 640), (129, 256, 1152),
    (513, 128, 3456), (1023, 128, 7168), (2047, 128, 16384),
])
def test_fine_bpreshuffle_signed_partitions(kid, shape):
    _check_signed_repeated(kid, shape, 4)


@pytest.mark.parametrize("kid", range(9060, 9070))
def test_fine_bpreshuffle_cancellation(kid):
    _check_signed_repeated(kid, (97, 256, 1280), 4, cancellation=True)


def _fine_data(m=97, n=256, k=640):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
    return tune, tune.generate_data(m, n, k, 29, device="cuda")


@pytest.mark.parametrize("kid", range(9062, 9070))
def test_fine_bpreshuffle_workspace_guards(kid):
    from aiter.ops.opus import opus_gemm
    tune, data = _fine_data()
    m, k = data["x"].shape
    n = data["out"].shape[1]
    count = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid].bpreshuffle_split_k * m * n
    storage = torch.full((count + 256,), 73, dtype=torch.float32, device="cuda")
    workspace = storage[128:-128]
    reference = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
    for _ in range(4):
        workspace.fill_(float("nan"))
        data["out"].fill_(float("nan"))
        result = opus_gemm(data["x"], data["w"], data["out"], kid=kid,
                           layout="bpreshuffle", x_scale=data["x_scale"],
                           w_scale=data["w_scale"], workspace=workspace)
        assert tune.compare_outputs(reference, result, printLog=False) == 0
        assert torch.isfinite(workspace).all()
        assert (storage[:128] == 73).all() and (storage[-128:] == 73).all()


@pytest.mark.parametrize("case", ["capacity", "dtype", "stride", "alignment", "alias"])
def test_fine_bpreshuffle_rejects_invalid_workspace(case):
    from aiter.ops.opus.gemm_op_a8w8 import _opus_gemm_bpreshuffle_workspace_raw
    _, data = _fine_data(17, 128, 128)
    count = 4 * 17 * 128
    if case == "capacity":
        workspace = torch.empty(count - 1, device="cuda", dtype=torch.float32)
    elif case == "dtype":
        workspace = torch.empty(count, device="cuda", dtype=torch.bfloat16)
    elif case == "stride":
        workspace = torch.empty(count * 2, device="cuda", dtype=torch.float32)[::2]
    elif case == "alignment":
        workspace = torch.empty(count + 1, device="cuda", dtype=torch.float32)[1:]
    else:
        workspace = torch.empty(count, device="cuda", dtype=torch.float32)
        data["out"] = workspace.view(torch.bfloat16)[:17 * 128].view(17, 128)
    with pytest.raises(RuntimeError, match="workspace"):
        _opus_gemm_bpreshuffle_workspace_raw(
            data["x"], data["w"], data["x_scale"], data["w_scale"],
            data["out"], 9063, workspace,
        )


@pytest.mark.parametrize("caller_workspace", [False, True])
def test_fine_bpreshuffle_graph_replay(caller_workspace):
    from aiter.ops.opus import opus_gemm
    tune, data = _fine_data(81, 256, 7168)
    kwargs = dict(kid=9063, layout="bpreshuffle", x_scale=data["x_scale"], w_scale=data["w_scale"])
    if caller_workspace:
        kwargs["workspace"] = torch.empty(4 * 81 * 256, device="cuda", dtype=torch.float32)
    for _ in range(2):
        opus_gemm(data["x"], data["w"], data["out"], **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        opus_gemm(data["x"], data["w"], data["out"], **kwargs)
    for _ in range(3):
        data["x"].copy_((-data["x"].float()).to(data["x"].dtype))
        reference = tune.run_torch(*(data[key] for key in tune._REF_KEYS), with_bounds=True)
        data["out"].fill_(float("nan"))
        graph.replay()
        assert tune.compare_outputs(reference, data["out"], printLog=False) == 0


@pytest.mark.parametrize("kid,shape", [(9000, (256, 256, 256)), (9010, (64, 256, 384))])
def test_fine_bpreshuffle_preserves_original_raw_launch(kid, shape):
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid", range(9070, 9074))
@pytest.mark.parametrize("shape", [(1, 128, 7168), (17, 256, 7168), (511, 384, 7168)])
def test_register_tail_bpreshuffle_signed_tails(kid, shape):
    # N=128/256 leaves two different N16 tails for the N48 candidates;
    # N=384 also exercises tiles crossing native N128 scale boundaries.
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid", range(9070, 9074))
def test_register_tail_bpreshuffle_cancellation(kid):
    _check_signed_repeated(kid, (129, 256, 7168), 3, cancellation=True)


@pytest.mark.parametrize("kid", range(9070, 9074))
def test_register_tail_bpreshuffle_shape_contract(kid):
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bpreshuffle_supports_shape
    instance = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
    for shape in ((1, 128, 7168), (17, 256, 7168), (512, 384, 7168)):
        assert a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)
    for shape in ((0, 128, 7168), (513, 128, 7168), (17, 192, 7168), (17, 128, 128)):
        assert not a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)


@pytest.mark.parametrize("kid", range(9070, 9074))
@pytest.mark.parametrize("shape,message", [((17, 128, 128), "requires K == 7168"),
                                           ((513, 128, 7168), "requires M <= 512")])
def test_register_tail_bpreshuffle_launcher_rejects_shape(kid, shape, message):
    tune, data = _fine_data(*shape)
    tune._ensure_kids_compiled({kid})
    with pytest.raises(RuntimeError, match=message):
        tune.run_bench(*(data[key] for key in tune._BENCH_KEYS), kid)


@pytest.mark.parametrize("kid,shape", [
    (9041, (17, 128, 7168)), (9041, (32, 2048, 7168)),
    (9041, (17, 128, 8192)),
    (9042, (17, 128, 7168)), (9042, (511, 384, 7168)),
    (9042, (97, 128, 7296)),
    (9044, (127, 7168, 1536)), (9044, (129, 7168, 1536)),
    (9052, (17, 128, 7168)), (9053, (97, 128, 7168)),
    (9062, (160, 128, 16384)), (9062, (161, 128, 16384)),
    (9063, (47, 128, 8192)), (9063, (49, 128, 8192)),
    (9063, (81, 128, 8192)), (9063, (97, 128, 8192)),
    (9063, (113, 128, 8192)), (9063, (129, 128, 8192)),
    (9063, (319, 2048, 7168)), (9063, (321, 2048, 7168)),
])
def test_consolidated_bpreshuffle_signed_dispatch(kid, shape):
    # Exercise both sides of dispatch boundaries with partial tiles. The
    # independent reference also checks that a choice uses the right grid,
    # LDS allocation, scale groups, and (where needed) FP32 workspace.
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid,shape", [
    (9042, (33, 256, 7168)), (9052, (33, 256, 7168)),
    (9053, (33, 256, 7168)), (9062, (177, 128, 8192)),
    (9063, (97, 128, 8192)), (9063, (113, 128, 8192)),
    (9063, (353, 2048, 7168)),
])
def test_consolidated_bpreshuffle_cancellation(kid, shape):
    _check_signed_repeated(kid, shape, 3, cancellation=True)
