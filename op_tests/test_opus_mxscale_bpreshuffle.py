import pytest
import torch

from csrc.opus_gemm.opus_gemm_common import (
    BpreshuffleConfig,
    a8w8_mxscale_bpreshuffle_supports_shape,
    a8w8_mxscale_gemm_bpreshuffle_kernels_list,
    config_id,
)


def _kid(pipeline, bm, bn, wm, wn, **parameters):
    return config_id(BpreshuffleConfig(pipeline, bm, bn, wm, wn, **parameters))


def _register_kid(bm, bn, wave_k, prefetch, **parameters):
    parameters = {"output": 3, "b_cache": 3, **parameters}
    return _kid("register", bm, bn, 1, 1, wave_k=wave_k,
                prefetch=prefetch, **parameters)


def _lds_kid(bm, bn, wm, wn, *, stages=4, **parameters):
    return _kid("lds_ring", bm, bn, wm, wn, stages=stages, **parameters)


def _fine_kid(bm, wm, wn, split_k=1, **parameters):
    return _lds_kid(bm, 128, wm, wn, split_k=split_k,
                    early_scale_loads=True, prefetch_before_read=True,
                    read_only_drain=True, fine_m_loads=True, **parameters)


def _panel_kid(*, scale_panel=128, **parameters):
    return _kid("panel_8wave", 192, 256, 4, 2,
                scale_panel=scale_panel, **parameters)


def _packed_kid(bn=128, *, stages=3, scale_panel=32, **parameters):
    return _kid("packed_4wave", 64, bn, 2, 2, stages=stages,
                scale_panel=scale_panel, **parameters)


def _compact_lds_kid(bm=160, *, stages=2, **parameters):
    return _kid("lds_4wave", bm, 128, 2, 2, stages=stages, **parameters)


def _quadrant_kid(pad_m=False):
    return _kid("quadrant_4wave", 256, 256, 2, 2,
                scale_panel=64, pad_m=pad_m)


_SMALL_REGISTER_KIDS = (
    _register_kid(16, 32, 1, 6),
    _register_kid(16, 16, 8, 2, output=0, b_cache=0),
    _register_kid(32, 32, 4, 3, b_cache=0),
    _register_kid(16, 16, 8, 3),
    _register_kid(16, 32, 4, 3),
    _register_kid(16, 32, 8, 2),
    _register_kid(32, 32, 8, 2),
    _register_kid(32, 64, 4, 2),
)
_SMALL_LDS_KIDS = (
    _lds_kid(32, 64, 1, 4, stages=8, cluster=2, early_scale_loads=True),
    _lds_kid(64, 64, 2, 2, early_scale_loads=True,
             prefetch_before_read=True, read_only_drain=True),
    _lds_kid(96, 64, 2, 2, output=1, read_only_drain=True),
    _lds_kid(64, 128, 4, 2, stages=6, cluster=2),
    _lds_kid(32, 64, 2, 2, register_scales=True),
    _lds_kid(32, 64, 2, 2, register_scales=True, xor_lds=True),
    _lds_kid(32, 128, 2, 2, cluster=2, register_scales=True,
             xor_lds=True, prefetch_before_read=True),
    _lds_kid(32, 64, 1, 4, stages=12, cluster=4, early_scale_loads=True),
    _lds_kid(64, 64, 2, 2, stages=8, cluster=2, early_scale_loads=True),
)
_FINE_KIDS = (
    _fine_kid(80, 1, 4),
    _fine_kid(96, 2, 4),
    _fine_kid(80, 1, 4, 2),
    _fine_kid(80, 1, 4, 4, reduce_vec=16, store_cache=2),
    _fine_kid(96, 2, 4, 2, reduce_vec=16, store_cache=2),
    _fine_kid(96, 2, 4, 4, reduce_vec=16, store_cache=2),
    _fine_kid(96, 2, 2, 4, reduce_vec=16, reduce_block=64),
    _fine_kid(128, 2, 2, 4),
    _fine_kid(112, 1, 4, 4),
    _fine_kid(48, 1, 4, 4),
)
_REGISTER_TAIL_KIDS = (
    _register_kid(16, 32, 8, 3, output=4, fixed_k=7168, reuse_b_scale=True),
    _register_kid(16, 48, 4, 4, output=4, fixed_k=7168, pad_n=True),
    _register_kid(32, 32, 4, 4, output=4, fixed_k=7168, reuse_b_scale=True),
    _register_kid(32, 48, 4, 3, output=4, b_cache=0, fixed_k=7168, pad_n=True),
)
_FIXED_PARTITION_KIDS = (
    *(_fine_kid(64, 1, 4, split_k, fixed_k=7168) for split_k in (1, 4, 8)),
    _fine_kid(48, 1, 4, 8, fixed_k=7168),
)


def test_grouped_bpreshuffle_contract():
    kernels = a8w8_mxscale_gemm_bpreshuffle_kernels_list
    instance = kernels[_panel_kid()]
    assert instance.bpreshuffle_config.pipeline == "panel_8wave"
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
    _check_signed_repeated(_panel_kid(), shape, 3)


def test_small_bpreshuffle_registry():
    from csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune import (
        candidate_kids_for_shape,
    )

    kernels = a8w8_mxscale_gemm_bpreshuffle_kernels_list
    assert len({instance.name for instance in kernels.values()}) == len(kernels)
    for kid in (*_SMALL_REGISTER_KIDS, *_SMALL_LDS_KIDS):
        instance = kernels[kid]
        assert instance.direct_only and instance.scale_dtype == "e8m0"
        assert instance.max_m is None and instance.max_k == 16384
        assert kid in candidate_kids_for_shape(
            "gfx950", 2049, 256, 128, all_candidates=True,
        )
        for shape in ((513, 256, 128), (2049, 256, 128)):
            assert a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)
        for shape in ((0, 256, 128), (16, 128, 16512), (16, 128, 129)):
            assert not a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)


@pytest.mark.parametrize("kid", [
    _register_kid(16, 16, 8, 2, output=0, b_cache=0),
    _register_kid(32, 32, 4, 3, b_cache=0),
    _lds_kid(32, 64, 1, 4, stages=8, cluster=2, early_scale_loads=True),
    _lds_kid(64, 64, 2, 2, early_scale_loads=True,
             prefetch_before_read=True, read_only_drain=True),
    _lds_kid(64, 128, 4, 2, stages=6, cluster=2),
    _lds_kid(32, 64, 2, 2, register_scales=True),
    _lds_kid(32, 64, 2, 2, register_scales=True, xor_lds=True),
    _lds_kid(32, 128, 2, 2, cluster=2, register_scales=True,
             xor_lds=True, prefetch_before_read=True),
])
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


@pytest.mark.parametrize("kid", [
    _register_kid(16, 32, 1, 6),
    _register_kid(32, 32, 4, 3, b_cache=0),
    _lds_kid(32, 64, 1, 4, stages=8, cluster=2, early_scale_loads=True),
    _lds_kid(96, 64, 2, 2, output=1, read_only_drain=True),
    _lds_kid(64, 128, 4, 2, stages=6, cluster=2),
])
@pytest.mark.parametrize("shape", [(97, 768, 1152), (97, 768, 16384)])
def test_small_bpreshuffle_shared_paths(kid, shape):
    _check_signed_repeated(kid, shape, 16)


@pytest.mark.parametrize("kid", [
    _register_kid(16, 16, 8, 3),
    _register_kid(16, 32, 4, 3),
    _register_kid(16, 32, 8, 2),
    _register_kid(32, 32, 8, 2),
    _register_kid(32, 64, 4, 2),
    _lds_kid(32, 64, 1, 4, stages=12, cluster=4, early_scale_loads=True),
    _lds_kid(64, 64, 2, 2, stages=8, cluster=2, early_scale_loads=True),
])
@pytest.mark.parametrize(
    "shape", [(1, 128, 128), (33, 256, 640), (511, 256, 1152), (97, 768, 16384)]
)
def test_optimized_bpreshuffle_signed_prefetch(kid, shape):
    _check_signed_repeated(kid, shape, 16)


@pytest.mark.parametrize("kid", [
    _lds_kid(32, 64, 1, 4, stages=8, cluster=2, early_scale_loads=True),
    _lds_kid(64, 64, 2, 2, early_scale_loads=True,
             prefetch_before_read=True, read_only_drain=True),
    _lds_kid(96, 64, 2, 2, output=1, read_only_drain=True),
    _lds_kid(64, 128, 4, 2, stages=6, cluster=2),
    _lds_kid(32, 64, 1, 4, stages=12, cluster=4, early_scale_loads=True),
    _lds_kid(64, 64, 2, 2, stages=8, cluster=2, early_scale_loads=True),
])
@pytest.mark.parametrize(
    "shape",
    [(513, 128, 128), (1023, 256, 640), (1729, 768, 7168),
     (2047, 128, 1152), (2048, 256, 16384)],
)
def test_extended_bpreshuffle_signed_tails(kid, shape):
    _check_signed_repeated(kid, shape, 16)


@pytest.mark.parametrize("kid", [
    _lds_kid(32, 64, 1, 4, stages=8, cluster=2, early_scale_loads=True),
    _lds_kid(64, 64, 2, 2, early_scale_loads=True,
             prefetch_before_read=True, read_only_drain=True),
    _lds_kid(96, 64, 2, 2, output=1, read_only_drain=True),
    _lds_kid(64, 128, 4, 2, stages=6, cluster=2),
    _register_kid(16, 16, 8, 3),
    _register_kid(16, 32, 4, 3),
    _register_kid(16, 32, 8, 2),
    _register_kid(32, 32, 8, 2),
    _register_kid(32, 64, 4, 2),
    _lds_kid(32, 64, 1, 4, stages=12, cluster=4, early_scale_loads=True),
    _lds_kid(64, 64, 2, 2, stages=8, cluster=2, early_scale_loads=True),
])
def test_optimized_bpreshuffle_cancellation(kid):
    config = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid].bpreshuffle_config
    m = 513 if config.pipeline == "lds_ring" and not config.register_scales else 97
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
    shape = ("gfx950", 256, 256, 768, 7168)
    rejected_kid = _quadrant_kid()
    accepted_kid = _compact_lds_kid(128, stages=3)
    rows = [
        ((shape, rejected_kid, 0, "", "opus", True), 1.0, 0.0001),
        ((shape, accepted_kid, 0, "", "opus", True), 3.0, 0.0),
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
    rejected_opus = next(row for row in saved if row["kernelId"] == str(rejected_kid))
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


@pytest.mark.parametrize("kid", _FINE_KIDS)
@pytest.mark.parametrize("shape", [
    (1, 128, 128), (81, 256, 640), (129, 256, 1152),
    (513, 128, 3456), (1023, 128, 7168), (2047, 128, 16384),
])
def test_fine_bpreshuffle_signed_partitions(kid, shape):
    _check_signed_repeated(kid, shape, 4)


@pytest.mark.parametrize("kid", _FINE_KIDS)
def test_fine_bpreshuffle_cancellation(kid):
    _check_signed_repeated(kid, (97, 256, 1280), 4, cancellation=True)


def _fine_data(m=97, n=256, k=640):
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    from csrc.opus_gemm import opus_gemm_mxscale_bpreshuffle_tune as tune
    return tune, tune.generate_data(m, n, k, 29, device="cuda")


@pytest.mark.parametrize("kid", [
    kid for kid in _FINE_KIDS
    if a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid].bpreshuffle_split_k > 1
])
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
            data["out"], _fine_kid(80, 1, 4, 4, reduce_vec=16, store_cache=2), workspace,
        )


@pytest.mark.parametrize("caller_workspace", [False, True])
def test_fine_bpreshuffle_graph_replay(caller_workspace):
    from aiter.ops.opus import opus_gemm
    tune, data = _fine_data(81, 256, 7168)
    kwargs = dict(kid=_fine_kid(80, 1, 4, 4, reduce_vec=16, store_cache=2),
                  layout="bpreshuffle", x_scale=data["x_scale"], w_scale=data["w_scale"])
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


@pytest.mark.parametrize("kid,shape", [
    (_quadrant_kid(), (256, 256, 256)),
    (_quadrant_kid(pad_m=True), (64, 256, 384)),
])
def test_quadrant_bpreshuffle_raw_launch(kid, shape):
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid", _REGISTER_TAIL_KIDS)
@pytest.mark.parametrize("shape", [(1, 128, 7168), (17, 256, 7168), (511, 384, 7168)])
def test_register_tail_bpreshuffle_signed_tails(kid, shape):
    # N=128/256 leaves two different N16 tails for the N48 candidates;
    # N=384 also exercises tiles crossing native N128 scale boundaries.
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid", _REGISTER_TAIL_KIDS)
def test_register_tail_bpreshuffle_cancellation(kid):
    _check_signed_repeated(kid, (129, 256, 7168), 3, cancellation=True)


@pytest.mark.parametrize("kid", _REGISTER_TAIL_KIDS)
def test_register_tail_bpreshuffle_shape_contract(kid):
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bpreshuffle_supports_shape
    instance = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
    for shape in ((1, 128, 7168), (17, 256, 7168), (512, 384, 7168), (513, 128, 7168)):
        assert a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)
    for shape in ((0, 128, 7168), (17, 192, 7168), (17, 128, 128)):
        assert not a8w8_mxscale_bpreshuffle_supports_shape(instance, *shape)


@pytest.mark.parametrize("kid", _REGISTER_TAIL_KIDS)
@pytest.mark.parametrize("shape,message", [((17, 128, 128), "requires K == 7168")])
def test_register_tail_bpreshuffle_launcher_rejects_shape(kid, shape, message):
    tune, data = _fine_data(*shape)
    tune._ensure_kids_compiled({kid})
    with pytest.raises(RuntimeError, match=message):
        tune.run_bench(*(data[key] for key in tune._BENCH_KEYS), kid)


@pytest.mark.parametrize("kid,shape", [
    (_register_kid(16, 16, 8, 3), (17, 128, 7168)),
    (_register_kid(16, 16, 8, 2, output=0, b_cache=0), (32, 2048, 7168)),
    (_register_kid(16, 16, 8, 3), (17, 128, 8192)),
    (_register_kid(16, 48, 4, 4, output=4, fixed_k=7168, pad_n=True), (17, 128, 7168)),
    (_register_kid(16, 48, 4, 4, output=4, fixed_k=7168, pad_n=True), (511, 384, 7168)),
    (_register_kid(32, 32, 4, 3, b_cache=0), (97, 128, 7296)),
    (_lds_kid(64, 64, 2, 2, stages=8, cluster=2, early_scale_loads=True), (127, 7168, 1536)),
    (_lds_kid(64, 64, 2, 2, early_scale_loads=True,
              prefetch_before_read=True, read_only_drain=True), (129, 7168, 1536)),
    (_register_kid(16, 32, 8, 3, output=4, fixed_k=7168, reuse_b_scale=True), (17, 128, 7168)),
    (_register_kid(32, 32, 4, 4, output=4, fixed_k=7168, reuse_b_scale=True), (97, 128, 7168)),
    (_fine_kid(80, 1, 4, 2, fixed_k=16384), (160, 128, 16384)),
    (_fine_kid(96, 2, 4, 2, reduce_vec=16, store_cache=2), (161, 128, 16384)),
    (_fine_kid(48, 1, 4, 4), (47, 128, 8192)),
    (_fine_kid(80, 1, 4, 4, reduce_vec=16, store_cache=2), (49, 128, 8192)),
    (_fine_kid(96, 2, 2, 4, reduce_vec=16, reduce_block=64), (81, 128, 8192)),
    (_fine_kid(112, 1, 4, 4), (97, 128, 8192)),
    (_fine_kid(128, 2, 2, 4), (113, 128, 8192)),
    (_fine_kid(80, 1, 4, 4, reduce_vec=16, store_cache=2), (129, 128, 8192)),
    (_fine_kid(80, 1, 4, 4, reduce_vec=16, store_cache=2), (319, 2048, 7168)),
    (_fine_kid(96, 2, 4, 4, reduce_vec=16, store_cache=2), (321, 2048, 7168)),
])
def test_bpreshuffle_direct_tile_and_schedule_tails(kid, shape):
    # Explicit configurations exercise partial tiles with their own grid,
    # LDS allocation, scale groups, and (where needed) FP32 workspace.
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid,shape", [
    (_register_kid(16, 48, 4, 4, output=4, fixed_k=7168, pad_n=True), (33, 256, 7168)),
    (_register_kid(16, 32, 8, 3, output=4, fixed_k=7168, reuse_b_scale=True), (33, 256, 7168)),
    (_register_kid(32, 32, 4, 4, output=4, fixed_k=7168, reuse_b_scale=True), (33, 256, 7168)),
    (_fine_kid(96, 2, 4, 2, reduce_vec=16, store_cache=2), (177, 128, 8192)),
    (_fine_kid(112, 1, 4, 4), (97, 128, 8192)),
    (_fine_kid(128, 2, 2, 4), (113, 128, 8192)),
    (_fine_kid(96, 2, 4, 4, reduce_vec=16, store_cache=2), (353, 2048, 7168)),
])
def test_bpreshuffle_direct_schedule_cancellation(kid, shape):
    _check_signed_repeated(kid, shape, 3, cancellation=True)


def test_fixedk_bpreshuffle_registry_contract():
    kernels = a8w8_mxscale_gemm_bpreshuffle_kernels_list
    assert len({instance.name for instance in kernels.values()}) == len(kernels)
    for kid in _FIXED_PARTITION_KIDS:
        assert a8w8_mxscale_bpreshuffle_supports_shape(kernels[kid], 193, 768, 7168)
        assert not a8w8_mxscale_bpreshuffle_supports_shape(kernels[kid], 193, 768, 7296)
        assert a8w8_mxscale_bpreshuffle_supports_shape(kernels[kid], 2049, 768, 7168)
    for stages, drain in ((4, False), (3, False), (4, True)):
        runtime = kernels[_lds_kid(32, 64, 2, 2, stages=stages,
                                   register_scales=True, read_only_drain=drain)]
        fixed = kernels[_lds_kid(32, 64, 2, 2, stages=stages, register_scales=True,
                                 read_only_drain=drain, fixed_k=768)]
        assert a8w8_mxscale_bpreshuffle_supports_shape(runtime, 513, 128, 1152)
        assert a8w8_mxscale_bpreshuffle_supports_shape(fixed, 513, 128, 768)
        assert not a8w8_mxscale_bpreshuffle_supports_shape(fixed, 177, 128, 1152)
    split = kernels[_fine_kid(80, 1, 4, 2, reduce_vec=16, fixed_k=16384)]
    assert split.bpreshuffle_split_k == 2
    assert a8w8_mxscale_bpreshuffle_supports_shape(split, 145, 768, 16384)
    assert not a8w8_mxscale_bpreshuffle_supports_shape(split, 145, 768, 8192)
    assert a8w8_mxscale_bpreshuffle_supports_shape(
        kernels[_packed_kid(fixed_k=16384)], 545, 768, 16384)
    compact = kernels[_compact_lds_kid(fixed_k=3072)]
    assert a8w8_mxscale_bpreshuffle_supports_shape(compact, 1040, 768, 3072)
    assert not a8w8_mxscale_bpreshuffle_supports_shape(compact, 1041, 768, 3072)


@pytest.mark.parametrize("kid", _FIXED_PARTITION_KIDS)
@pytest.mark.parametrize("shape", ((193, 128, 7168), (385, 768, 7168)))
def test_fixedk_bpreshuffle_narrow_k_partitions(kid, shape):
    # K=7168 / split8 gives seven K128 tiles: a partial ring/drain plus M tails.
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("stages,drain", ((4, False), (3, False), (4, True)))
@pytest.mark.parametrize("fixed_k,shape", (
    (384, (17, 128, 384)), (768, (145, 768, 768)),
    (1024, (177, 128, 1024)), (0, (33, 256, 640)),
))
def test_fixedk_bpreshuffle_shortk_drain(stages, drain, fixed_k, shape):
    # Distinct A scales and partial M test that a read-only drain never consumes
    # a queue slot before its register scale and matrix operands are complete.
    kid = _lds_kid(32, 64, 2, 2, stages=stages, register_scales=True,
                   read_only_drain=drain, fixed_k=fixed_k)
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid,shape", (
    (_fine_kid(80, 1, 4, fixed_k=7168), (1023, 128, 7168)),
    (_fine_kid(80, 1, 4, 4, fixed_k=16384), (2047, 128, 16384)),
    (_fine_kid(48, 1, 4, 4, stages=6, cluster=2, fixed_k=7168), (1023, 128, 7168)),
    (_fine_kid(80, 1, 4, 2, reduce_vec=16, fixed_k=16384), (145, 128, 16384)),
    (_packed_kid(fixed_k=7168), (545, 128, 7168)),
    (_packed_kid(fixed_k=16384), (577, 128, 16384)),
    (_packed_kid(), (545, 128, 1536)),
    (_compact_lds_kid(fixed_k=384), (144, 128, 384)),
    (_compact_lds_kid(fixed_k=768), (1040, 128, 768)),
    (_compact_lds_kid(fixed_k=1024), (1040, 128, 1024)),
    (_compact_lds_kid(fixed_k=1536), (144, 128, 1536)),
    (_compact_lds_kid(fixed_k=3072), (1040, 128, 3072)),
    (_compact_lds_kid(fixed_k=7168), (1552, 128, 7168)),
    (_compact_lds_kid(fixed_k=16384), (1040, 128, 16384)),
))
def test_fixedk_bpreshuffle_scale_panel_refill(kid, shape):
    # Long FixedK still refills SP32; it is not limited to the first scale panel.
    _check_signed_repeated(kid, shape, 3)


@pytest.mark.parametrize("kid,shape", (
    (_fine_kid(64, 1, 4, 8, fixed_k=7168), (193, 128, 7168)),
    (_fine_kid(48, 1, 4, 8, fixed_k=7168), (193, 128, 7168)),
    (_lds_kid(32, 64, 2, 2, register_scales=True,
              read_only_drain=True, fixed_k=768), (145, 128, 768)),
    (_fine_kid(80, 1, 4, 2, reduce_vec=16, fixed_k=16384), (145, 128, 16384)),
    (_packed_kid(fixed_k=7168), (545, 128, 7168)),
    (_compact_lds_kid(fixed_k=7168), (1040, 128, 7168)),
))
def test_fixedk_bpreshuffle_cancellation(kid, shape):
    _check_signed_repeated(kid, shape, 3, cancellation=True)
