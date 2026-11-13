# Opus a4w4_mxfp4_nt_noshuffle (bpreshuffle=False) vs the bpreshuffle opus path and aiter gemm_a4w4 asm; CUDA graph of IT
# calls, median of 5 replays.  Same random data for all (fp4 bytes, e8m0 in [124, 130]); the shuffled copies are derived
# from it.  Portable: only needs this aiter checkout on PYTHONPATH (JIT builds module_deepgemm_opus).
# usage: AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=<aiter checkout> HIP_VISIBLE_DEVICES=<gpu> python3 perf_noshuffle.py [M,N,K ...]
# prints per shape: asm, opus bpreshuffle dispatch, opus noshuffle dispatch, every noshuffle kid.
import sys, torch, aiter
from aiter.ops.gemm_op_a4w4 import gemm_a4w4
from aiter.ops.opus import gemm_a4w4_opus, opus_gemm_a4w4_tune
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle

IT = 100; dev = "cuda"; fp4 = aiter.dtypes.fp4x2; e8 = aiter.dtypes.fp8_e8m0
T = ("256x256", "256x128", "128x256", "128x128", "192x256", "256x192")
NS_KIDS = {**{30100 + i: t + " pers" for i, t in enumerate(T)}, **{30110 + i: t + " np" for i, t in enumerate(T)}, 30130: "256x256 nob"}
DEFAULT = [f"{m},{n},{k}" for (n, k) in ((4096, 4096), (2048, 7168), (3072, 6144), (4096, 8192))
           for m in (128, 256, 512, 1024, 2048, 4096)] + ["8192,8192,8192"]


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
    if N % 16 == 0:
        wsh, xss, wss = shuffle_weight(w, layout=(16, 16)), e8m0_shuffle(xs), e8m0_shuffle(ws)
        try:
            ya = torch.empty(((M + 31) // 32 * 32, N), dtype=torch.bfloat16, device=dev)
            res["asm"] = graph_us(lambda: gemm_a4w4(x, wsh, xss, wss, bpreshuffle=True))
        except Exception as e:  # noqa: BLE001
            res["asm"] = float("nan"); print(f"# asm failed: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
        res["opus_shuffle"] = graph_us(lambda: gemm_a4w4_opus(x, wsh, xss, wss, Y=y))
    res["opus_noshuffle"] = graph_us(lambda: gemm_a4w4_opus(x, w, xs, ws, Y=y, bpreshuffle=False))
    for kid, t in NS_KIDS.items():
        if kid == 30130 and (M % 256 or N % 256):
            continue
        res[t] = graph_us(lambda: opus_gemm_a4w4_tune(x, w, xs, ws, Y=y, kernelId=kid))
    best = min((v, k) for k, v in res.items() if k in NS_KIDS.values())
    print(f"{M} {N} {K} | " + " ".join(f"{k}={v:.2f}" for k, v in res.items()) + f" | best_ns={best[1]} {best[0]:.2f}", flush=True)
