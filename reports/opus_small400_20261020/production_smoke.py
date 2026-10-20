"""Check real default-table routing, packed scale bytes and caller output."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
from report_environment import configure
configure()
os.environ['ROCR_VISIBLE_DEVICES'] = '4'
os.environ['AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_MXSCALE_BPRESHUFFLE'] = str(
    ROOT / 'aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv')
for name in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
    os.environ.pop(name, None)
sys.path.insert(0, str(ROOT))
import torch
import aiter
from aiter.utility import dtypes
aiter.dtypes = dtypes
from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_blockscale_bpreshuffle, get_mxscale_bpreshuffle_config)
from aiter.ops.opus import gemm_op_a8w8 as native
from csrc.gemm_a8w8_mxfp8_bpreshuffle import gemm_a8w8_mxfp8_bpreshuffle_tune as t
from csrc.opus_gemm.opus_gemm_common import (
    a8w8_mxscale_gemm_bpreshuffle_kernels_list as registry,
    a8w8_mxscale_bpreshuffle_supports_shape as supports)

torch.set_num_threads(4)
results = []
cases = [(32, 768, 7168), (2, 768, 7168), (480, 16384, 1536),
         (1, 7168, 384), (33, 768, 7168), (1041, 7168, 384),
         (1088, 6144, 7168), (1, 768, 7168), (1088, 768, 7168)]
for m, n, k in cases:
    row = get_mxscale_bpreshuffle_config(m, n, k, '128x128')
    data = t.generate_data(m, n, k, 197, '128x128')
    expected = torch.nn.functional.linear(data['x_deq'], data['w_deq'])
    error = 1e-4 + 5e-5 * torch.nn.functional.linear(data['x_deq'].abs(), data['w_deq'].abs())
    lower = (expected - error).to(dtypes.bf16).float()
    upper = (expected + error).to(dtypes.bf16).float()
    with patch.object(native, '_launch_a8w8_blockscale_bpreshuffle_gemm',
                      wraps=native._launch_a8w8_blockscale_bpreshuffle_gemm) as launch:
        for byte_scales in (False, True):
            xs, ws = data['x_scale'], data['w_scale']
            if byte_scales:
                xs, ws = xs.view(torch.uint8), ws.view(torch.uint8)
            for caller_out in (False, True):
                data['out'].fill_(float('nan'))
                y = gemm_a8w8_blockscale_bpreshuffle(
                    data['x'], data['weight_shuffle'], xs, ws,
                    out=data['out'] if caller_out else None)
                if caller_out:
                    assert y.data_ptr() == data['out'].data_ptr()
                assert torch.isfinite(y).all().item()
                assert ((y.float() >= lower) & (y.float() <= upper)).all().item()
        uses_opus = row['libtype'] == 'opus' and supports(
            registry[int(row['kernelId'])], m, n, k)
        if uses_opus:
            assert launch.call_count == 4
            assert all(c.kwargs['kid'] == int(row['kernelId']) for c in launch.call_args_list)
        else:
            assert launch.call_count == 0
    result = dict(shape=[m, n, k], backend='opus' if uses_opus else 'flydsl',
                  table_backend=row['libtype'], kid=str(row['kernelId']),
                  checks=4, errors=0)
    results.append(result)
    print(json.dumps(result), flush=True)
(P / 'production_smoke.json').write_text(json.dumps(results, indent=2) + '\n')
