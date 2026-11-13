# gemm_a4w4_opus_noshuffle (row-major inputs, the Triton gemm_afp4wfp4 layout) vs Triton gemm_afp4wfp4 on the same inputs
# (asm is not compared: it needs e8m0_shuffle'd scales, a different layout); CUDA graph of IT calls, median of 5 replays.  Same random data for all (fp4 bytes, e8m0 in [124, 130]); the shuffled copies are derived
# from it.  Portable: only needs this aiter checkout on PYTHONPATH (JIT builds module_deepgemm_opus).
# usage: AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=<aiter checkout> HIP_VISIBLE_DEVICES=<gpu> python3 perf_noshuffle.py [M,N,K ...]
# prints per shape: triton, opus noshuffle dispatch, ours/triton, every noshuffle kid (KIDS=0 skips the per-kid sweep).
import sys, torch, aiter
import os
from aiter.ops.opus import gemm_a4w4_opus_noshuffle
try:
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
except Exception:  # noqa: BLE001
    gemm_afp4wfp4 = None

IT = 100; dev = "cuda"; fp4 = aiter.dtypes.fp4x2; e8 = aiter.dtypes.fp8_e8m0
T = ("256x256", "256x128", "128x256", "128x128", "192x256", "256x192")
NS_KIDS = {**{30100 + i: t + " pers" for i, t in enumerate(T)}, **{30110 + i: t + " np" for i, t in enumerate(T)}, 30130: "256x256 nob"}
DEFAULT = [f"{m},{n},{k}" for (n, k) in ((4096, 4096), (2048, 7168), (3072, 6144), (4096, 8192))
           for m in (1, 16, 64, 128, 256, 512, 1024, 2048, 4096)] + ["8192,8192,8192"]


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


for spec in sys.argv[1:] or DEFAULT:
    M, N, K = (int(v) for v in spec.split(","))
    torch.manual_seed(0)
    x = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev).view(fp4)
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev).view(fp4)
    xs = torch.randint(124, 131, (M, K // 32), dtype=torch.uint8, device=dev).view(e8)
    ws = torch.randint(124, 131, (N, K // 32), dtype=torch.uint8, device=dev).view(e8)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    res = {}
    xu, wu, xsu, wsu = x.view(torch.uint8), w.view(torch.uint8), xs.view(torch.uint8), ws.view(torch.uint8)
    if gemm_afp4wfp4 is not None:
        try:
            res["triton"] = graph_us(lambda: gemm_afp4wfp4(xu, wu, xsu, wsu, torch.bfloat16, y))
        except Exception as e:  # noqa: BLE001
            res["triton"] = float("nan"); print(f"# triton failed: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
    res["opus_noshuffle"] = graph_us(lambda: gemm_a4w4_opus_noshuffle(xu, wu, xsu, wsu, Y=y))
    res["ours/triton"] = res["opus_noshuffle"] / res["triton"] if res.get("triton") else float("nan")
    for kid, t in (NS_KIDS.items() if os.environ.get("KIDS", "1") != "0" else ()):
        if kid == 30130 and (M % 256 or N % 256):
            continue
        res[t] = graph_us(lambda: gemm_a4w4_opus_noshuffle(xu, wu, xsu, wsu, Y=y, kernelId=kid))
    kids = [(v, k) for k, v in res.items() if k in NS_KIDS.values()]
    best = f" | best_ns={min(kids)[1]} {min(kids)[0]:.2f}" if kids else ""
    print(f"{M} {N} {K} | " + " ".join(f"{k}={v:.3f}" if k == "ours/triton" else f"{k}={v:.2f}" for k, v in res.items()) + best, flush=True)
