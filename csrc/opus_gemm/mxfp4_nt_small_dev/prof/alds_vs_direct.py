#!/usr/bin/env python3
# Experiment: can A_LDS (BM 16 / in-block K split KS) replace B_DIRECT at small M (shuffled inputs)?
# Times the B_DIRECT kid configs (30400-30405), the A_LDS small-M candidates and aiter asm on one GPU, same way as
# perf_all.py (hipGraph of 100 launches, median of 5); every shape guarded by an idle check, each config timed twice in
# alternating order (min kept).
#   python3 alds_vs_direct.py --small-exe /tmp/alds_ks/small.exe --gpu 0 --smi 3 [--out DIR]
# The B_DIRECT shuffle configs need a harness built from 1777ae898 (shuffle B_DIRECT was removed after this experiment).
import argparse, csv, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_all import ASM_PY, sh, wait_idle  # noqa: E402

DIRECT = [("30400 dir 16x32 ks4 d3", "-a 0 -t 16x32 -s 4 -d 3"), ("30401 dir 16x32 ks4 d4", "-a 0 -t 16x32 -s 4 -d 4"),
          ("30402 dir 32x32 ks4 d2", "-a 0 -t 32x32 -s 4 -d 2"), ("30403 dir 32x64 ks4 d2", "-a 0 -t 32x64 -s 4 -d 2"),
          ("30404 dir 32x64 ks2 d3", "-a 0 -t 32x64 -s 2 -d 3"), ("30405 dir 16x64 ks4 d3", "-a 0 -t 16x64 -s 4 -d 3")]
ALDS = [(f"alds {t} ks{s} d{d}", f"-a 2 -t {t} -s {s} -d {d}") for t, s, ds in [
    ("16x32", 4, (3, 4, 6, 8)), ("16x32", 2, (4, 6, 8)), ("16x64", 4, (3, 4, 5)), ("16x64", 2, (4, 6)),
    ("16x64", 1, (4, 6, 8)), ("16x128", 1, (4, 6)), ("32x32", 4, (3, 4, 6)), ("32x64", 4, (3, 4)), ("32x64", 2, (4, 6)),
    ("32x64", 1, (6, 8))] for d in ds]
NK = [(512, 4096), (2048, 7168), (4096, 4096), (7168, 2048), (3072, 1536), (8192, 1024)]
MS = [1, 8, 16, 32]


def t_small(exe, gpu, flags, M, N, K):
    out = sh(f"flock /tmp/gpu{gpu}.lock env HIP_VISIBLE_DEVICES={gpu} {exe} {flags} -p 1 -m {M} -n {N} -k {K} -b 1 -v 0 -i 100")
    m = re.search(r"time ([0-9.]+) us", out)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small-exe", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--smi", type=int, required=True)
    ap.add_argument("--out", default="./alds_vs_direct_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    asm_py = os.path.join(a.out, "asm_one.py"); open(asm_py, "w").write(ASM_PY)
    cfgs = DIRECT + ALDS
    rows = []
    for N, K in NK:
        for M in MS:
            wait_idle(a.smi, 1800)
            res = {}
            for rep in range(2):
                for label, flags in (cfgs if rep == 0 else cfgs[::-1]):
                    t = t_small(a.small_exe, a.gpu, flags, M, N, K)
                    if t is not None:
                        res[label] = min(res.get(label, t), t)
            m = re.search(r"ASM \d+ \d+ \d+ ([0-9.]+) (\S+)",
                          sh(f"flock /tmp/gpu{a.gpu}.lock env HIP_VISIBLE_DEVICES={a.gpu} python3 {asm_py} {M},{N},{K}", timeout=1200))
            asm = float(m.group(1)) if m else None
            busy_after = not wait_idle(a.smi, 0)
            for k, v in res.items():
                rows.append({"M": M, "N": N, "K": K, "config": k, "us": v, "asm_us": asm, "busy_after": int(busy_after)})
            bd = min((v, k) for k, v in res.items() if k.startswith("304"))
            ba = min((v, k) for k, v in res.items() if k.startswith("alds"))
            print(f"M={M:2d} N={N:5d} K={K:5d}  direct {bd[0]:7.2f} ({bd[1][:5]})  alds {ba[0]:7.2f} ({ba[1]})  "
                  f"alds/direct {ba[0] / bd[0]:.3f}  asm {asm}  {'BUSY' if busy_after else ''}", flush=True)
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
