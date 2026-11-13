# Opus a4w4_mxfp4_nt vs aiter gemm_a4w4 asm, CUDA graph of IT calls, median of 5 replays (same method as work_smallm/asm_small.py).
# usage: AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=<worktree> python3 perf_vs_asm.py "M,N,K" ...
# prints per shape: asm, opus dispatch (heuristic), and every opus kid (split-K kids for S in 2/4/8).
import os, sys, torch, aiter
from aiter.ops.gemm_op_a4w4 import gemm_a4w4, get_GEMM_config
from aiter.ops.opus import gemm_a4w4_opus, opus_gemm_a4w4_tune
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle
IT = 100; dev = 'cuda'; fp4 = aiter.dtypes.fp4x2; e8 = aiter.dtypes.fp8_e8m0
T = ("256x256", "256x128", "128x256", "128x128", "192x256", "256x192")
KIDS = {**{30000 + i: t + " pers" for i, t in enumerate(T)}, **{30010 + i: t + " np" for i, t in enumerate(T)},
        **{30020 + i: t + " sk" for i, t in enumerate(T) if i}, 30030: "256x256 nob"}


def graph_us(f):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(10): f()          # eager warm-up on the capture stream (allocates the split-K workspace)
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


for spec in sys.argv[1:]:
    M, N, K = map(int, spec.split(',')); torch.manual_seed(0)
    x = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev); w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    xs = torch.randint(124, 131, (M, K // 32), dtype=torch.uint8, device=dev); ws = torch.randint(124, 131, (N, K // 32), dtype=torch.uint8, device=dev)
    xq = x.view(fp4); wsh = shuffle_weight(w.view(fp4), layout=(16, 16)); xss = e8m0_shuffle(xs.view(e8)); wss = e8m0_shuffle(ws.view(e8))
    res = {"asm": graph_us(lambda: gemm_a4w4(xq, wsh, xss, wss, bpreshuffle=True))}
    y = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    res["opus"] = graph_us(lambda: gemm_a4w4_opus(xq, wsh, xss, wss, Y=y))
    for kid, t in KIDS.items():
        if kid == 30030 and (M % 256 or N % 256):
            continue
        if kid >= 30020 and kid < 30030:
            for S in (2, 4, 8):
                if K % (S * 256):
                    continue
                os.environ["OPUS_MXFP4_SPLITK"] = str(S)
                res[f"{t}{S}"] = graph_us(lambda: opus_gemm_a4w4_tune(xq, wsh, xss, wss, Y=y, kernelId=kid))
                del os.environ["OPUS_MXFP4_SPLITK"]
        else:
            res[t] = graph_us(lambda: opus_gemm_a4w4_tune(xq, wsh, xss, wss, Y=y, kernelId=kid))
    best = min((v, k) for k, v in res.items() if k not in ("asm", "opus"))
    name = (get_GEMM_config(M, N, K) or {}).get('kernelName', 'default')
    print(f"{M} {N} {K} | asm {res['asm']:.2f} | opus(dispatch) {res['opus']:.2f} | best {best[1]} {best[0]:.2f} | "
          + " ".join(f"{k}={v:.2f}" for k, v in res.items() if k not in ('asm', 'opus')) + f" | asm_kernel {name}", flush=True)
