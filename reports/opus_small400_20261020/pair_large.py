"""Pair retained large kernels against the remote build using identical ABI calls."""
import csv
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
from report_environment import configure
configure()
os.environ['ROCR_VISIBLE_DEVICES'] = '4'
for name in ('HIP_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES'):
    os.environ.pop(name, None)
sys.path.insert(0, str(ROOT))
import torch
import aiter
from aiter.utility import dtypes
aiter.dtypes = dtypes
from aiter.jit.core import get_module
from aiter.test_common import run_perftest
from csrc.gemm_a8w8_mxfp8_bpreshuffle import gemm_a8w8_mxfp8_bpreshuffle_tune as t

baseline_path = Path(sys.argv[1])
current = get_module('module_deepgemm_opus')
sys.path.insert(0, str(baseline_path.parent))
spec = importlib.util.spec_from_file_location('remote.module_deepgemm_opus', baseline_path)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
modules = {'remote': baseline, 'current': current}
torch.set_num_threads(4)

def run_descriptor(x, w, xs, ws, out, kid, tag):
    module = modules[tag]
    tensors = [dtypes.torch_to_aiter_pybind(v) for v in (x.unsqueeze(0), w.unsqueeze(0), xs, ws, out.unsqueeze(0))]
    module._set_current_hip_stream(torch.cuda.current_stream().cuda_stream)
    module.opus_gemm_a8w8_blockscale_bpreshuffle_launch(*tensors, kid)
    return out

rows = []
for s in json.loads((P / 'large294_plan.json').read_text()):
    r = json.loads((P / 'regression294' / ('_'.join(map(str, s['shape'])) + '.json')).read_text())
    us = next(iter(r['medians'].values()))
    if us <= s['baseline_us'] * 1.05 or not s['kids']:
        continue
    m, n, k = s['shape']
    kid = s['kids'][0]
    data = t.generate_data(m, n, k, 211, '128x128')
    expected = torch.nn.functional.linear(data['x_deq'], data['w_deq'])
    bound = 1e-4 + 5e-5 * torch.nn.functional.linear(data['x_deq'].abs(), data['w_deq'].abs())
    lower = (expected - bound).to(dtypes.bf16).float()
    upper = (expected + bound).to(dtypes.bf16).float()
    values = [data[q] for q in ('x', 'weight_shuffle', 'x_scale_opus', 'w_scale', 'out')]
    times = {'remote': [], 'current': []}
    for round_id in range(3):
        choices = [('remote', baseline), ('current', current)]
        if round_id % 2:
            choices.reverse()
        for tag, module in choices:
            data['out'].fill_(float('nan'))
            y, latency = run_perftest(run_descriptor, *values, kid, tag,
                                     num_warmup=5, num_iters=101)
            assert torch.isfinite(y).all().item()
            assert ((y.float() >= lower) & (y.float() <= upper)).all().item()
            times[tag].append(float(latency))
    entry = dict(M=m, N=n, K=k, kernelId=kid,
                 remote_us=statistics.median(times['remote']),
                 current_us=statistics.median(times['current']),
                 rounds=times, checks=6, errors=0)
    entry['current_over_remote'] = entry['current_us'] / entry['remote_us']
    rows.append(entry)
    print(json.dumps(entry), flush=True)
    (P / 'large_pair.json').write_text(json.dumps(rows, indent=2) + '\n')
    del data, values, y, expected, bound, lower, upper
    torch.cuda.empty_cache()
