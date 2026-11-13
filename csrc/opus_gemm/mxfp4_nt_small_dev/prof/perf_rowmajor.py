#!/usr/bin/env python3
# Row-major (Triton gemm_afp4wfp4 layout) sweep: small-tile pipeline (-l 1, every config, split-K 1/2/4) vs Triton
# gemm_afp4wfp4 (config=None) vs the noshuffle 2x2 pipeline (gemm_a4w4_opus_noshuffle, every kid) -- same GPU, idle checks.
#   AITER_USE_SYSTEM_TRITON=1 PYTHONPATH=<aiter checkout> python3 perf_rowmajor.py --exe <small.exe> --gpu <HIP idx> --out DIR
# Triton / 2x2 come from ../../mxfp4_nt_noshuffle_dev/perf/perf_noshuffle.py (CUDA graph, 100 calls, median of 5); the
# small-tile exe times a hipGraph of 100 launches, median of 5.  Writes results.csv, select.csv, table.txt.
import argparse, csv, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perf_all import sh, smi_index, busy, wait_idle   # noqa: E402

CFG = [("direct 16x32 ks4 d2", "-a 0 -t 16x32 -s 4 -d 2"), ("direct 16x32 ks4 d4", "-a 0 -t 16x32 -s 4 -d 4"),
       ("direct 16x64 ks4 d2", "-a 0 -t 16x64 -s 4 -d 2"), ("direct 16x64 ks4 d4", "-a 0 -t 16x64 -s 4 -d 4"),
       ("direct 32x32 ks4 d2", "-a 0 -t 32x32 -s 4 -d 2"), ("direct 32x32 ks4 d4", "-a 0 -t 32x32 -s 4 -d 4"),
       ("direct 32x64 ks4 d2", "-a 0 -t 32x64 -s 4 -d 2"), ("direct 32x64 ks2 d2", "-a 0 -t 32x64 -s 2 -d 2"),
       ("lds 16x64 d3", "-a 1 -t 16x64 -d 3"), ("lds 16x64 d4", "-a 1 -t 16x64 -d 4"),
       ("lds 32x64 d3", "-a 1 -t 32x64 -d 3"), ("lds 32x64 d4", "-a 1 -t 32x64 -d 4"),
       ("lds 64x64 d3", "-a 1 -t 64x64 -d 3"), ("lds 16x128 d4", "-a 1 -t 16x128 -d 4"),
       ("lds 32x128 d3", "-a 1 -t 32x128 -d 3"), ("lds 32x128 d4", "-a 1 -t 32x128 -d 4"),
       ("lds 64x128 d3", "-a 1 -t 64x128 -d 3"), ("lds 64x128 d4", "-a 1 -t 64x128 -d 4"),
       ("lds 32x256 d3", "-a 1 -t 32x256 -d 3"), ("lds 64x256 d2", "-a 1 -t 64x256 -d 2"),
       ("lds 128x128 d2", "-a 1 -t 128x128 -d 2"), ("lds 128x128 d3", "-a 1 -t 128x128 -d 3")]
NK = [(4096, 4096), (2048, 7168), (3072, 6144), (4096, 8192)]
MS = [1, 16, 64, 128, 256, 512]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True); ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--smi", type=int, default=-1); ap.add_argument("--out", default="./perf_rm")
    ap.add_argument("--shapes", default=""); ap.add_argument("--max-wait", type=int, default=1800)
    a = ap.parse_args()
    smi = a.smi if a.smi >= 0 else smi_index(a.gpu)
    here = os.path.dirname(os.path.abspath(__file__))
    ns_py = os.path.normpath(os.path.join(here, "../../mxfp4_nt_noshuffle_dev/perf/perf_noshuffle.py"))
    os.makedirs(a.out, exist_ok=True)
    shapes = [tuple(map(int, s.split(","))) for s in a.shapes.split(";")] if a.shapes else [(m, n, k) for n, k in NK for m in MS]
    print(f"HIP {a.gpu} = smi {smi}", flush=True)
    rows, sel = [], []
    for M, N, K in shapes:
        for attempt in range(3):
            if not wait_idle(smi, a.max_wait):
                print(f"{M} {N} {K}: GPU never idle", flush=True); break
            res = {}
            for lab, fl in CFG:
                for S in (1, 2, 4):
                    if S > 1 and K % (S * 256):
                        continue
                    o = sh(f"HIP_VISIBLE_DEVICES={a.gpu} {a.exe} -l 1 {fl} -p {S} -m {M} -n {N} -k {K} -b 1 -v 0 -i 100")
                    m = re.search(r"time ([0-9.]+) us", o)
                    if m:
                        res[f"small {lab}" + (f" sk{S}" if S > 1 else "")] = float(m.group(1))
            o = sh(f"HIP_VISIBLE_DEVICES={a.gpu} python3 {ns_py} {M},{N},{K}", timeout=1800)
            tri = re.search(r"triton=([0-9.]+)", o)
            big = {f"big {k}": float(v) for k, v in re.findall(r"(\d+x\d+ (?:pers|np|nob))=([0-9.]+)", o)}
            res.update(big)
            if not busy(smi):
                break
            print(f"{M} {N} {K}: disturbed, retry", flush=True)
        t = float(tri.group(1)) if tri else None
        for k, v in res.items():
            rows.append({"M": M, "N": N, "K": K, "config": k, "us": v, "triton_us": t})
        bs = min(((v, k) for k, v in res.items() if k.startswith("small")), default=(None, "-"))
        bb = min(((v, k) for k, v in res.items() if k.startswith("big")), default=(None, "-"))
        best = min([x for x in (bs, bb) if x[0]], default=(None, "-"))
        r = lambda x: f"{x / t:.3f}" if (x and t) else "-"
        sel.append({"M": M, "N": N, "K": K, "best": best[1], "us": best[0], "triton_us": t, "ratio": r(best[0]),
                    "best_small": bs[1], "small_us": bs[0], "best_2x2": bb[1], "big_us": bb[0]})
        print(f"{M:4d} {N:5d} {K:5d} | triton {t} | small {bs[0]} ({bs[1]}) ratio {r(bs[0])} | 2x2 {bb[0]} ({bb[1]}) ratio {r(bb[0])}", flush=True)
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["M", "N", "K", "config", "us", "triton_us"]); w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "select.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sel[0].keys())); w.writeheader(); w.writerows(sel)


if __name__ == "__main__":
    main()
