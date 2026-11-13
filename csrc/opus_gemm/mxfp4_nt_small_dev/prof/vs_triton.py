#!/usr/bin/env python3
# Cross-check the small-tile pipeline (row-major layout) against aiter Triton gemm_afp4wfp4 on identical inputs.
#   python3 vs_triton.py EXE "flags" "M,N,K" ...   (EXE run with -l 1 -D <tmp> -b 1; GPU from HIP_VISIBLE_DEVICES)
import os, subprocess, sys, tempfile, numpy as np, torch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
exe, flags = sys.argv[1], sys.argv[2]
for spec in sys.argv[3:]:
    M, N, K = map(int, spec.split(","))
    d = tempfile.mkdtemp()
    out = subprocess.run(f"{exe} -l 1 {flags} -m {M} -n {N} -k {K} -b 1 -v 0 -D {d}", shell=True, capture_output=True, text=True)
    ld = lambda f, shape: torch.from_numpy(np.fromfile(os.path.join(d, f), dtype=np.uint8).reshape(shape)).cuda()
    a, b = ld("a.bin", (M, K // 2)), ld("b.bin", (N, K // 2))
    sa, sb = ld("sfa.bin", (M, K // 32)), ld("sfb.bin", (N, K // 32))
    ours = torch.from_numpy(np.fromfile(os.path.join(d, "c.bin"), dtype=np.uint16).reshape(M, N).view(np.int16)).cuda().view(torch.bfloat16).float()
    tri = gemm_afp4wfp4(a, b, sa, sb, torch.bfloat16).float()
    rel = ((ours - tri).norm() / tri.norm()).item()
    nan = int(torch.isnan(ours).sum())
    print(f"{M} {N} {K} {flags}: rel_diff_vs_triton {rel:.3e}  max_abs {((ours - tri).abs().max().item()):.3e}  nan {nan}  {'OK' if rel < 1e-2 and nan == 0 else 'MISMATCH'}", flush=True)
