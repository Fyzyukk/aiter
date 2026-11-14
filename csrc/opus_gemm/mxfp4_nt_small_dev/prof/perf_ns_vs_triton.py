#!/usr/bin/env python3
# No-shuffle (row-major, Triton gemm_afp4wfp4 layout) GEMMs vs Triton on identical inputs, all through torch on one GPU:
#   - the no-shuffle 2x2 big kids (30100-30105 persistent, 30110-30115 one tile per block, 30130 256x256 no boundary),
#   - the no-shuffle small-tile kids (A4W4_MXFP4_NT_SMALL_ROWMAJOR_KIDS),
#   - aiter Triton gemm_afp4wfp4 (config=None).
# Every kernel is checked against a dequantised fp32 reference before it is timed (CUDA graph of IT calls, median of 5
# replays, as perf_noshuffle.py).  Writes one CSV row per (shape, kernel).
#   AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=<aiter checkout> HIP_VISIBLE_DEVICES=<gpu> python3 perf_ns_vs_triton.py OUT.csv [M,N,K ...]
import csv, sys, torch, aiter
from aiter.ops.opus import gemm_a4w4_opus_noshuffle
from aiter.ops.opus.gemm_op_a4w4 import A4W4_MXFP4_NT_SMALL_ROWMAJOR_KIDS
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

IT = 100; dev = "cuda"
BIG = {**{30100 + i: t + " pers" for i, t in enumerate(("256x256", "256x128", "128x256", "128x128", "192x256", "256x192"))},
       **{30110 + i: t + " np" for i, t in enumerate(("256x256", "256x128", "128x256", "128x128", "192x256", "256x192"))},
       30130: "256x256 nob"}
NK = ((4096, 4096), (2048, 7168), (3072, 6144), (4096, 8192))
DEFAULT = [f"{m},{n},{k}" for (n, k) in NK for m in (1, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192)]
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=dev)


def deq(q, s):
    v = torch.stack([LUT[(q & 15).long()], LUT[(q >> 4).long()]], -1).reshape(q.shape[0], -1)
    return v * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)


def graph_us(f):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(10): f()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(IT): f()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); r = []
    for _ in range(5):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); r.append(e0.elapsed_time(e1) * 1e3 / IT)
    return sorted(r)[2]


out = open(sys.argv[1], "w", newline=""); w = csv.writer(out); w.writerow(["M", "N", "K", "kernel", "us", "rel_err"])
for spec in sys.argv[2:] or DEFAULT:
    M, N, K = (int(v) for v in spec.split(","))
    torch.manual_seed(0)
    x = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
    wq = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    xs = torch.randint(124, 131, (M, K // 32), dtype=torch.uint8, device=dev)
    ws = torch.randint(124, 131, (N, K // 32), dtype=torch.uint8, device=dev)
    ref = deq(x, xs) @ deq(wq, ws).T
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    runs = {"triton": lambda: gemm_afp4wfp4(x, wq, xs, ws, torch.bfloat16, y)}
    for kid in list(BIG) + list(A4W4_MXFP4_NT_SMALL_ROWMAJOR_KIDS):
        if kid == 30130 and (M % 256 or N % 256):
            continue
        runs[f"kid{kid}"] = (lambda kid=kid: gemm_a4w4_opus_noshuffle(x, wq, xs, ws, Y=y, kernelId=kid))
    line = []
    for name, f in runs.items():
        try:
            y.fill_(float("nan")); f(); torch.cuda.synchronize()
            rel = ((y.float() - ref).norm() / ref.norm()).item()
            ok = rel < 1e-2 and not torch.isnan(y).any().item()
            us = graph_us(f) if ok else float("nan")
        except Exception as e:  # noqa: BLE001
            rel, us = float("nan"), float("nan"); print(f"# {M},{N},{K} {name}: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
        w.writerow([M, N, K, name, f"{us:.3f}", f"{rel:.2e}"]); line.append((us, name))
    out.flush()
    ok = [(u, n) for u, n in line if u == u]
    print(f"{M} {N} {K} best {min(ok)} triton {dict((n, u) for u, n in line)['triton']:.3f}", flush=True)
