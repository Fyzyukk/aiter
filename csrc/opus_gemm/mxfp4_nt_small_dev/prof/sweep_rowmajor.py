#!/usr/bin/env python3
# Row-major small-tile config sweep (how the 30500-30504 configs were chosen): every pick_rm config of the harness
# (S 1; S 2 / 4 for M <= 64) on M x 4 (N, K), idle-checked per shape; writes the per-(shape, config) minimum over rounds.
#   python3 sweep_rowmajor.py --exe <small.exe> --gpu <HIP idx> --out sweep.csv [--rounds R] [--ms 1,8,16,...]
import argparse, csv, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from perf_all import sh, smi_index, busy, wait_idle   # noqa: E402
src = open(os.path.join(HERE, "..", "gemm_a4w4_mxfp4_nt_small_host.cc")).read()
body = src[src.index("launch_fn pick_rm"):src.index("#undef R")]
CFGS = [tuple(map(int, m)) for m in re.findall(r"R\((\d+), (\d+), (\d+), (\d+)\)", body)][1:]   # skip the #define line
ap = argparse.ArgumentParser(); ap.add_argument("--exe"); ap.add_argument("--gpu", type=int); ap.add_argument("--out"); ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--ms", default="1,8,16,64,128,256,512,1024")
a = ap.parse_args(); smi = smi_index(a.gpu)
NK = [(4096, 4096), (2048, 7168), (3072, 6144), (4096, 8192)]
best = {}
for r in range(a.rounds):
    for M in map(int, a.ms.split(",")):
        for N, K in NK:
            for att in range(4):
                wait_idle(smi, 1800); res = {}
                for bm, bn, ks, d in CFGS:
                    for S in ((1, 2, 4) if M <= 64 else (1,)):
                        o = sh(f"HIP_VISIBLE_DEVICES={a.gpu} {a.exe} -l 1 -t {bm}x{bn} -s {ks} -d {d} -p {S} -m {M} -n {N} -k {K} -v 0 -i 100")
                        m = re.search(r"time ([0-9.]+) us", o)
                        if m: res[(bm, bn, ks, d, S)] = float(m.group(1))
                if not busy(smi): break
                print("disturbed", M, N, K, flush=True)
            for k, v in res.items():
                key = (M, N, K) + k; best[key] = min(best.get(key, 1e9), v)
            b = min(res.items(), key=lambda x: x[1])
            print(f"r{r} {M} {N} {K} best {b}", flush=True)
            with open(a.out, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["M", "N", "K", "BM", "BN", "KS", "D", "S", "us"])
                for k, v in sorted(best.items()): w.writerow(list(k) + [v])
