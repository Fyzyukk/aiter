# aiter.gemm_a4w4(bpreshuffle=True): asm default vs the opus backend (AITER_A4W4_BACKEND=opus, opus dispatch), plus Triton
# gemm_afp4wfp4 (row-major inputs, same data) for reference.  CUDA graph of IT calls, median of 5 replays.
# usage: AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=<aiter checkout> HIP_VISIBLE_DEVICES=<gpu> python3 perf_gemm_a4w4.py [M,N,K ...]
import os, sys, torch, aiter
from aiter.ops.gemm_op_a4w4 import gemm_a4w4
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle

IT = 100; dev = "cuda"; fp4 = aiter.dtypes.fp4x2; e8 = aiter.dtypes.fp8_e8m0
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


def backend(name, f):
    """AITER_A4W4_BACKEND is read on every gemm_a4w4 call, so it can be toggled per measurement (graph capture included)."""
    if name: os.environ["AITER_A4W4_BACKEND"] = name
    else: os.environ.pop("AITER_A4W4_BACKEND", None)
    try:
        return graph_us(f)
    finally:
        os.environ.pop("AITER_A4W4_BACKEND", None)


try:
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
except Exception:  # noqa: BLE001
    gemm_afp4wfp4 = None
print("# M N K | asm (gemm_a4w4 default) | opus (gemm_a4w4, AITER_A4W4_BACKEND=opus) | triton gemm_afp4wfp4 | opus/asm")
for spec in sys.argv[1:] or DEFAULT:
    M, N, K = (int(v) for v in spec.split(","))
    torch.manual_seed(0)
    x = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    xs = torch.randint(124, 131, (M, K // 32), dtype=torch.uint8, device=dev)
    ws = torch.randint(124, 131, (N, K // 32), dtype=torch.uint8, device=dev)
    wsh, xss, wss = shuffle_weight(w.view(fp4), layout=(16, 16)), e8m0_shuffle(xs.view(e8)), e8m0_shuffle(ws.view(e8))
    f = lambda: gemm_a4w4(x.view(fp4), wsh, xss, wss, bpreshuffle=True)
    asm, opus = backend("", f), backend("opus", f)
    tri = float("nan")
    if gemm_afp4wfp4 is not None:
        y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
        try:
            tri = graph_us(lambda: gemm_afp4wfp4(x, w, xs, ws, torch.bfloat16, y))
        except Exception as e:  # noqa: BLE001
            print(f"# triton failed {M},{N},{K}: {type(e).__name__}: {str(e)[:80]}", file=sys.stderr)
    print(f"{M} {N} {K} | asm={asm:.2f} opus={opus:.2f} triton={tri:.2f} | {opus / asm:.3f}", flush=True)
