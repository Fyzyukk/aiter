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


def test_experimental_bpreshuffle_registry_contract():
    from csrc.opus_gemm.opus_gemm_common import A8W8_BPRESHUFFLE_EXPERIMENTAL_KIDS

    kernels = a8w8_mxscale_gemm_bpreshuffle_kernels_list
    experimental = A8W8_BPRESHUFFLE_EXPERIMENTAL_KIDS
    assert experimental.isdisjoint(A8W8_BPRESHUFFLE_TUNING_KIDS)
    assert experimental == frozenset(range(9080, 9090))
    assert len({instance.name for instance in kernels.values()}) == len(kernels)
    for kid in (9080, 9081, 9082, 9083):
        assert a8w8_mxscale_bpreshuffle_supports_shape(kernels[kid], 193, 768, 7168)
        assert not a8w8_mxscale_bpreshuffle_supports_shape(kernels[kid], 193, 768, 7296)
        assert not a8w8_mxscale_bpreshuffle_supports_shape(kernels[kid], 2049, 768, 7168)
    for kid in (9084, 9085, 9086):
        assert a8w8_mxscale_bpreshuffle_supports_shape(kernels[kid], 177, 128, 768)
        assert not a8w8_mxscale_bpreshuffle_supports_shape(kernels[kid], 513, 128, 768)
        assert not a8w8_mxscale_bpreshuffle_supports_shape(kernels[kid], 177, 128, 1152)
    assert kernels[9082].bpreshuffle_split_k == kernels[9083].bpreshuffle_split_k == 8
    assert kernels[9087].bpreshuffle_split_k == 2
    assert a8w8_mxscale_bpreshuffle_supports_shape(kernels[9087], 145, 768, 16384)
    assert not a8w8_mxscale_bpreshuffle_supports_shape(kernels[9087], 145, 768, 8192)
    assert a8w8_mxscale_bpreshuffle_supports_shape(kernels[9088], 545, 768, 16384)
    assert a8w8_mxscale_bpreshuffle_supports_shape(kernels[9089], 1040, 768, 3072)
    assert not a8w8_mxscale_bpreshuffle_supports_shape(kernels[9089], 1041, 768, 3072)


@pytest.mark.parametrize("kid", (9080, 9081, 9082, 9083))
@pytest.mark.parametrize("shape", ((193, 128, 7168), (385, 768, 7168)))
def test_experimental_bpreshuffle_narrow_k_partitions(kid, shape):
    # K=7168 / split8 gives seven K128 tiles: a partial ring/drain plus M tails.
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid", (9084, 9085, 9086))
@pytest.mark.parametrize("shape", (
    (17, 128, 384), (145, 768, 768), (177, 128, 1024), (33, 256, 640),
))
def test_experimental_bpreshuffle_shortk_drain(kid, shape):
    # Distinct A scales and partial M test that a read-only drain never consumes
    # a queue slot before its register scale and matrix operands are complete.
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid,shape", (
    (9087, (145, 128, 16384)),
    (9088, (545, 128, 7168)), (9088, (577, 128, 16384)),
    (9088, (545, 128, 1536)),
    (9089, (144, 128, 384)), (9089, (1040, 128, 768)),
    (9089, (1040, 128, 1024)), (9089, (144, 128, 1536)),
    (9089, (1040, 128, 3072)),
    (9089, (1552, 128, 7168)), (9089, (1040, 128, 16384)),
))
def test_experimental_bpreshuffle_fixedk_scale_panel_refill(kid, shape):
    # Long FixedK still refills SP32; it is not limited to the first scale panel.
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid,shape", (
    (9082, (193, 128, 7168)), (9083, (193, 128, 7168)),
    (9086, (145, 128, 768)), (9087, (145, 128, 16384)),
    (9088, (545, 128, 7168)), (9089, (1040, 128, 7168)),
))
def test_experimental_bpreshuffle_cancellation(kid, shape):
    _check_signed_repeated(kid, shape, 3, cancellation=True)
