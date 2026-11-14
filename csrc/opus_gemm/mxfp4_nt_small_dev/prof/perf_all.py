#!/usr/bin/env python3
# Portable one-command performance sweep: small-tile pipeline (shuffle 1x4 kid configs) + optional 2x2 big-tile pipeline
# + aiter gemm_a4w4 asm, same GPU, every shape guarded by "GPU idle before and after" checks.
#
#   python3 perf_all.py --small-exe /tmp/opus_mxfp4_small/small.exe [--big-exe <mxfp4_nt_dev exe>] --gpu 2 [--out DIR]
#
# --gpu is the HIP_VISIBLE_DEVICES index; the matching amd-smi / rocm-smi index (used for the idle check) is found by PCI
# bus id (they differ on some machines), or set --smi explicitly.  asm needs `import aiter` to work (PYTHONPATH / install;
# set AITER_USE_SYSTEM_TRITON=1 if aiter complains about triton).  All kernels are timed the same way: a graph / tight loop
# of back-to-back launches, median of 5 replays.
# Outputs in --out: results.csv (every config x shape), table.txt (best per family vs asm), select.csv ((M,N,K) -> best).
import argparse, csv, os, re, subprocess, sys, time

SHAPES_NK = [(4096, 4096), (2048, 7168), (3072, 6144), (4096, 8192), (4096, 1024), (3072, 1536)]
SHAPES_M = [1, 16, 64, 128, 256, 512, 1024]
# small-tile configs: (label, flags)
SMALL = [("1x4 16x32 ks2 d4", "-t 16x32 -s 2 -d 4"), ("1x4 16x32 ks4 d4", "-t 16x32 -s 4 -d 4"),
         ("1x4 16x64 d4", "-t 16x64 -d 4"),
         ("1x4 32x64 d6", "-t 32x64 -d 6"), ("1x4 32x128 d3", "-t 32x128 -d 3"),
         ("1x4 32x128 d4", "-t 32x128 -d 4"), ("1x4 64x128 d4", "-t 64x128 -d 4")]
SPLITS = [2, 4]   # cross-block split-K tried on top of every small-tile config when K allows it
# 2x2 big-tile configs (mxfp4_nt_dev host: -t tile, -p mode 0 persistent / 1 one tile per block / 2 split-K, -s S)
BIG_TILES = ["256x256", "256x128", "128x256", "128x128", "192x256", "256x192"]

ASM_PY = r'''
import sys, torch, aiter
from aiter.ops.gemm_op_a4w4 import gemm_a4w4, get_GEMM_config
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle
IT=100; dev='cuda'; fp4=aiter.dtypes.fp4x2; e8=aiter.dtypes.fp8_e8m0
for spec in sys.argv[1:]:
    M,N,K=map(int,spec.split(',')); torch.manual_seed(0)
    x=torch.randint(0,256,(M,K//2),dtype=torch.uint8,device=dev); w=torch.randint(0,256,(N,K//2),dtype=torch.uint8,device=dev)
    xs=torch.randint(124,131,(M,K//32),dtype=torch.uint8,device=dev); ws=torch.randint(124,131,(N,K//32),dtype=torch.uint8,device=dev)
    xq=x.view(fp4); wsh=shuffle_weight(w.view(fp4),layout=(16,16)); xss=e8m0_shuffle(xs.view(e8)); wss=e8m0_shuffle(ws.view(e8))
    f=lambda: gemm_a4w4(xq,wsh,xss,wss,bpreshuffle=True)
    for _ in range(20): f()
    torch.cuda.synchronize(); s=torch.cuda.Stream()
    with torch.cuda.stream(s):
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(IT): f()
    for _ in range(5): g.replay()
    torch.cuda.synchronize(); r=[]
    for _ in range(5):
        e0=torch.cuda.Event(enable_timing=True); e1=torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(); r.append(e0.elapsed_time(e1)*1e3/IT)
    cfg=get_GEMM_config(M,N,K); name=(cfg or {}).get('kernelName','default')
    print(f"ASM {M} {N} {K} {sorted(r)[2]:.3f} {name}", flush=True)
'''


def sh(cmd, env=None, timeout=600):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return ""


def smi_index(hip):
    """amd-smi / rocm-smi index of HIP device `hip`, matched by PCI bus id."""
    out = sh(f"HIP_VISIBLE_DEVICES={hip} python3 -c \"import torch; p=torch.cuda.get_device_properties(0); print(p.pci_bus_id)\"")
    try:
        bus = int(out.strip().split()[-1])
    except (ValueError, IndexError):
        return hip
    for line in sh("rocm-smi --showbus").splitlines():
        m = re.search(r"GPU\[(\d+)\].*PCI Bus: [0-9A-Fa-f]+:([0-9A-Fa-f]+):", line)
        if m and int(m.group(2), 16) == bus:
            return int(m.group(1))
    return hip


def busy(smi):
    occ = re.findall(r"CU_OCCUPANCY: (\d+)", sh(f"amd-smi process -g {smi}"))
    use = re.findall(r"GPU use \(%\): (\d+)", sh(f"rocm-smi -d {smi} --showuse"))
    return any(int(o) > 0 for o in occ) or (use and int(use[0]) > 5)


def wait_idle(smi, max_wait):
    t0 = time.time()
    while busy(smi):
        if time.time() - t0 > max_wait:
            return False
        time.sleep(5)
    return True


def time_small(exe, gpu, flags, M, N, K, split):
    out = sh(f"HIP_VISIBLE_DEVICES={gpu} {exe} {flags} -p {split} -m {M} -n {N} -k {K} -b 1 -v 0 -i 100")
    m = re.search(r"time ([0-9.]+) us", out)
    return float(m.group(1)) if m else None


def time_big(exe, gpu, tile, mode, split, M, N, K):
    out = sh(f"HIP_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=16 {exe} -t {tile} -p {mode} -s {split} -m {M} -n {N} -k {K} -b 1 -v 0 -w 200 -i 100")
    m = re.search(r"\(([0-9.]+) us\)", out)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-exe", required=True)
    ap.add_argument("--big-exe", default="")
    ap.add_argument("--gpu", type=int, required=True, help="HIP_VISIBLE_DEVICES index")
    ap.add_argument("--smi", type=int, default=-1, help="amd-smi index (default: matched by PCI bus)")
    ap.add_argument("--out", default="./perf_out")
    ap.add_argument("--max-wait", type=int, default=1800, help="seconds to wait for an idle GPU per shape")
    ap.add_argument("--no-asm", action="store_true")
    ap.add_argument("--shapes", default="", help="'M,N,K;M,N,K;...' (default: target set)")
    a = ap.parse_args()
    smi = a.smi if a.smi >= 0 else smi_index(a.gpu)
    os.makedirs(a.out, exist_ok=True)
    shapes = [tuple(map(int, s.split(","))) for s in a.shapes.split(";")] if a.shapes else \
             [(m, n, k) for (n, k) in SHAPES_NK for m in SHAPES_M]
    asm_py = os.path.join(a.out, "asm_one.py"); open(asm_py, "w").write(ASM_PY)
    print(f"GPU HIP {a.gpu} = smi {smi}; {len(shapes)} shapes -> {a.out}", flush=True)
    rows = []
    for (M, N, K) in shapes:
        for attempt in range(3):
            if not wait_idle(smi, a.max_wait):
                print(f"{M} {N} {K}: GPU never idle, skipped", flush=True); break
            res = {}
            for label, flags in SMALL:
                res["small " + label] = time_small(a.small_exe, a.gpu, flags, M, N, K, 1)
                for S in SPLITS:
                    if K % (S * 256) == 0:
                        res[f"small {label} sk{S}"] = time_small(a.small_exe, a.gpu, flags, M, N, K, S)
            if a.big_exe:
                for t in BIG_TILES:
                    res[f"big {t} pers"] = time_big(a.big_exe, a.gpu, t, 0, 1, M, N, K)
                    res[f"big {t} np"] = time_big(a.big_exe, a.gpu, t, 1, 1, M, N, K)
                    for S in SPLITS:
                        if t != "256x256" and K % (S * 256) == 0:
                            res[f"big {t} sk{S}"] = time_big(a.big_exe, a.gpu, t, 2, S, M, N, K)
            asm, asm_k = None, ""
            if not a.no_asm:
                m = re.search(r"ASM \d+ \d+ \d+ ([0-9.]+) (\S+)", sh(f"HIP_VISIBLE_DEVICES={a.gpu} python3 {asm_py} {M},{N},{K}", timeout=1200))
                if m:
                    asm, asm_k = float(m.group(1)), m.group(2)
            if not busy(smi):
                break
            print(f"{M} {N} {K}: GPU became busy during the run, retry {attempt + 1}", flush=True)
        for k, v in res.items():
            rows.append({"M": M, "N": N, "K": K, "config": k, "us": v, "asm_us": asm, "asm_kernel": asm_k})
        ok = {k: v for k, v in res.items() if v}
        bs = min(((v, k) for k, v in ok.items() if k.startswith("small")), default=(None, "-"))
        bb = min(((v, k) for k, v in ok.items() if k.startswith("big")), default=(None, "-"))
        fmt = lambda x: f"{x:.2f}" if x else "-"
        print(f"{M:5d} {N:5d} {K:5d} | asm {fmt(asm)} {asm_k} | small {fmt(bs[0])} {bs[1]} | big {fmt(bb[0])} {bb[1]}", flush=True)
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "N", "K", "config", "us", "asm_us", "asm_kernel"]); w.writeheader(); w.writerows(rows)
    sel, lines = [], []
    for (M, N, K) in shapes:
        r = [x for x in rows if (x["M"], x["N"], x["K"]) == (M, N, K) and x["us"]]
        if not r:
            continue
        b = min(r, key=lambda x: x["us"]); asm = r[0]["asm_us"]
        sel.append({"M": M, "N": N, "K": K, "pipeline": b["config"].split()[0], "config": b["config"], "us": b["us"],
                    "asm_us": asm, "ratio": round(b["us"] / asm, 3) if asm else ""})
        lines.append(f"{M:5d} {N:5d} {K:5d}  best {b['config']:28s} {b['us']:8.2f} us   asm {asm if asm else '-'}   ratio {sel[-1]['ratio']}")
    with open(os.path.join(a.out, "select.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "N", "K", "pipeline", "config", "us", "asm_us", "ratio"]); w.writeheader(); w.writerows(sel)
    open(os.path.join(a.out, "table.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
