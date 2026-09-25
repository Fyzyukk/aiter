"""The new exact IDs must reach the native E8M0 bpreshuffle launcher."""
import pytest
import torch
from aiter.ops.opus import opus_gemm
from aiter.ops.opus import gemm_op_a8w8 as implementation


@pytest.mark.parametrize("kid", range(9030, 9034))
def test_native_scales_reach_registered_launcher(monkeypatch, kid):
    calls = []
    monkeypatch.setattr(implementation, "_opus_gemm_a8w8_blockscale_bpreshuffle_launch_raw",
                        lambda *args: calls.append(args))
    x = torch.empty((64, 128), dtype=torch.float8_e4m3fn)
    w = torch.empty((256, 128), dtype=x.dtype)
    out = torch.empty((64, 256), dtype=torch.bfloat16)
    sa = torch.empty((1, 64), dtype=torch.uint8).T
    sb = torch.empty((2, 1), dtype=torch.uint8)
    assert opus_gemm(x, w, out, kid=kid, layout="bpreshuffle", x_scale=sa, w_scale=sb) is out
    assert len(calls) == 1
    args = calls[0]
    assert args[-1] == kid
    assert args[2].data_ptr() == sa.data_ptr() and args[3].data_ptr() == sb.data_ptr()
    assert args[2].dtype == sa.dtype and args[3].dtype == sb.dtype
    assert args[2].stride() == sa.stride() and args[3].stride() == sb.stride()
    assert args[0].data_ptr() == x.data_ptr() and args[1].data_ptr() == w.data_ptr()
    assert args[4].data_ptr() == out.data_ptr()
