# SPDX-License-Identifier: MIT
"""GPU regression checks for the gfx950 bpreshuffle workspace/Tensor ABI.

Build OPUS with kids 9502853, 9502872, 9502885, 9502939, 9503524, 9503619 and
9503624 before running this module.
"""
import pytest
import torch


@pytest.fixture(scope="module")
def api():
    if not torch.cuda.is_available():
        pytest.skip("requires gfx950")
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"):
        pytest.skip("requires gfx950")
    from aiter.ops.opus import opus_gemm
    from aiter.ops.opus.gemm_op_a8w8 import _opus_gemm_bpreshuffle_workspace_tensor_raw
    from csrc.opus_gemm.opus_gemm_common import (
        kernels_list, bpreshuffle_workspace_bytes, kernel_needs_external_workspace,
    )
    return (opus_gemm, _opus_gemm_bpreshuffle_workspace_tensor_raw,
            make_native_data, kernels_list, bpreshuffle_workspace_bytes,
            kernel_needs_external_workspace)


def make_native_data(m, n, k, seed):
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility import dtypes

    torch.manual_seed(seed)
    x = (torch.randn((m, k), device="cuda") * 2).to(dtypes.fp8)
    w = (torch.randn((n, k), device="cuda") * 2).to(dtypes.fp8)
    xs = torch.randint(124, 131, (m, k // 128), device="cuda", dtype=torch.uint8)
    ws = torch.randint(124, 131, (n // 128, k // 128), device="cuda", dtype=torch.uint8)
    x_deq = x.float() * torch.exp2(xs.float() - 127).repeat_interleave(128, 1)
    w_deq = w.float() * torch.exp2(ws.float() - 127).repeat_interleave(128, 0).repeat_interleave(128, 1)
    xs = xs.T.contiguous().view(m, k // 128).view(k // 128, m).T
    return dict(x=x, weight_shuffle=shuffle_weight(w, layout=(16, 16)),
                x_scale_opus=xs.view(dtypes.fp8_e8m0), w_scale=ws.view(dtypes.fp8_e8m0),
                out=torch.empty((m, n), device="cuda", dtype=dtypes.bf16),
                x_deq=x_deq, w_deq=w_deq)


def make_case(api, kid=9502885, m=3, n=128):
    _, _, generate_data, registry, workspace_bytes, _ = api
    inst = registry[kid]
    k = inst.bpreshuffle_fixed_k
    data = generate_data(m, n, k, 17)
    values = [data[key] for key in ("x", "weight_shuffle", "x_scale_opus", "w_scale", "out")]
    nbytes = workspace_bytes(inst, m, n, k)
    workspace = torch.zeros((nbytes + 3) // 4, device="cuda", dtype=torch.float32) if nbytes else None
    reference = (torch.nn.functional.linear(data["x_deq"], data["w_deq"]),
                 torch.nn.functional.linear(data["x_deq"].abs(), data["w_deq"].abs()))
    return values, workspace, reference


def check_result(reference, result):
    expected, magnitude = reference
    assert result.shape == expected.shape
    assert torch.isfinite(result).all().item()
    error = 1e-4 + 5e-5 * magnitude
    lower = (expected - error).to(result.dtype).float()
    upper = (expected + error).to(result.dtype).float()
    assert ((result.float() >= lower) & (result.float() <= upper)).all().item()


def invoke(api, values, kid, workspace=None, raw=False):
    if raw:
        return api[1](*values, kid, workspace)
    x, w, xs, ws, y = values
    return api[0](x, w, y, kid=kid, layout="bpreshuffle",
                  x_scale=xs, w_scale=ws, workspace=workspace)


@pytest.mark.parametrize("kid,m", [(9502885, 3), (9502885, 33), (9502885, 1041),
                                    (9502853, 48), (9502872, 3),
                                    (9503524, 3), (9503619, 33), (9503624, 1041)])
def test_generic_shapes_workspace_reuse_and_graph(api, kid, m):
    # These dimensions are outside the original 451-row selection. In
    # particular M=1041 checks that the new pipeline has no M<=1024 cutoff.
    values, workspace, reference = make_case(api, kid, m)
    needs_workspace = api[-1]("gfx950", "a8w8_blockscale_bpreshuffle", kid)
    assert needs_workspace == (workspace is not None)
    check_result(reference, invoke(api, values, kid))
    for _ in range(2):
        values[-1].fill_(float("nan"))
        check_result(reference, invoke(api, values, kid, workspace))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke(api, values, kid, workspace)
    for _ in range(3):
        values[-1].fill_(float("nan"))
        graph.replay()
        check_result(reference, values[-1])
        entry = api[3][kid].bpreshuffle_entry
        if entry.get("workspace_zero_init"):
            tiles = ((m + api[3][kid].B_M - 1) // api[3][kid].B_M) * (128 // api[3][kid].B_N)
            assert torch.count_nonzero(workspace.view(torch.int32)[-tiles:]).item() == 0


@pytest.mark.parametrize("kid,m", [(9502885, 3), (9502885, 33), (9502939, 48)])
def test_fused_split_overwrites_uninitialized_partials(api, kid, m):
    values, workspace, reference = make_case(api, kid, m)
    inst = api[3][kid]
    tiles = ((m + inst.B_M - 1) // inst.B_M) * (128 // inst.B_N)
    workspace.fill_(float("nan"))
    workspace[-tiles:].zero_()
    for _ in range(3):
        values[-1].fill_(float("nan"))
        check_result(reference, invoke(api, values, kid, workspace))
        assert torch.count_nonzero(workspace.view(torch.int32)[-tiles:]).item() == 0


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("case", ["x_dtype", "x_stride", "w_shape", "y_shape",
    "scale_layout", "scale_cpu", "ws_capacity", "ws_dtype", "ws_stride", "ws_alignment"])
def test_rejects_invalid_tensor_and_workspace(api, raw, case):
    values, workspace, _ = make_case(api)
    m, k = values[0].shape
    if case == "x_dtype":
        values[0] = values[0].float()
    elif case == "x_stride":
        values[0] = torch.empty((m, 2*k), device="cuda", dtype=values[0].dtype)[:, ::2]
    elif case == "w_shape":
        values[1] = values[1][:, :k//2].contiguous()
    elif case == "y_shape":
        values[4] = values[4][:, :64].contiguous()
    elif case == "scale_layout":
        values[2] = values[2].contiguous()
    elif case == "scale_cpu":
        values[3] = values[3].cpu()
    elif case == "ws_capacity":
        workspace = workspace[:-1]
    elif case == "ws_dtype":
        workspace = workspace.to(torch.bfloat16)
    elif case == "ws_stride":
        workspace = torch.empty(2*workspace.numel(), device="cuda")[::2]
    elif case == "ws_alignment":
        workspace = torch.empty(workspace.numel()+1, device="cuda")[1:]
    with pytest.raises((RuntimeError, ValueError, TypeError, AssertionError)):
        invoke(api, values, 9502885, workspace, raw)


def test_native_nondefault_stream_and_torch_compile(api):
    values, workspace, reference = make_case(api)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        values[-1].fill_(float("nan"))
        result = invoke(api, values, 9502885, workspace)
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    check_result(reference, result)
    native = api[1]

    def operation(x, w, xs, ws, y, scratch):
        return native(x, w, xs, ws, y, 9502885, scratch)

    compiled = torch.compile(operation, backend="eager", fullgraph=True)
    values[-1].fill_(float("nan"))
    check_result(reference, compiled(*values, workspace))


def test_byte_scales_and_automatic_workspace_graph(api):
    values, _, reference = make_case(api)
    values[2] = values[2].view(torch.uint8)
    values[3] = values[3].view(torch.uint8)
    check_result(reference, invoke(api, values, 9502885))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke(api, values, 9502885)
    for _ in range(3):
        values[-1].fill_(float("nan"))
        graph.replay()
        check_result(reference, values[-1])


def test_automatic_workspace_cache_is_stream_local_and_capture_safe(api):
    from aiter.ops.opus.gemm_op_a8w8 import _cached_bpreshuffle_workspace

    _cached_bpreshuffle_workspace.cache_clear()
    values, _, reference = make_case(api, m=33)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    outputs = []
    for stream in streams:
        output = torch.empty_like(values[-1])
        outputs.append(output)
        with torch.cuda.stream(stream):
            for _ in range(3):
                invoke(api, [*values[:-1], output], 9502885)
    for stream, output in zip(streams, outputs):
        stream.synchronize()
        check_result(reference, output)
    info = _cached_bpreshuffle_workspace.cache_info()
    assert info.misses == 2 and info.hits == 4
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke(api, values, 9502885)
    # Capture owns a separate allocation and does not retain it in the cache.
    assert _cached_bpreshuffle_workspace.cache_info() == info
    _cached_bpreshuffle_workspace.cache_clear()
    for _ in range(3):
        graph.replay()
        check_result(reference, values[-1])
