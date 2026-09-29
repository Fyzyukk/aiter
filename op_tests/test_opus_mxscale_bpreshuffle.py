import pytest
import torch

from csrc.opus_gemm.opus_gemm_common import (
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
    for kid in range(9040, 9050):
        instance = kernels[kid]
        assert instance.direct_only and instance.scale_dtype == "e8m0"
        assert instance.max_m == 512 and instance.max_k == 16384
        assert kid in candidate_kids_for_shape("gfx950", 511, 256, 1152)
        for shape in ((513, 256, 128), (16, 128, 16512), (16, 128, 129)):
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


def _check_signed_repeated(kid, shape, repetitions):
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
