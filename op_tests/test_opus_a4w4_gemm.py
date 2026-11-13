# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness / performance of the opus gfx950 MXFP4 GEMM (a4w4_mxfp4_nt) vs a torch reference and aiter gemm_a4w4 asm.

Inputs are random fp4 bytes and e8m0 scales in [124, 130] (the data used for kernel development); the reference
dequantises with the raw (unshuffled) operands. Run: python3 op_tests/test_opus_a4w4_gemm.py [--perf]
"""
import argparse
import sys

import torch

import aiter
from aiter.ops.opus import gemm_a4w4_opus, opus_gemm_a4w4_tune
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.fp4_utils import e8m0_shuffle

KIDS = {30000: "256x256", 30001: "256x128", 30002: "128x256", 30003: "128x128", 30004: "192x256", 30005: "256x192"}
_LUT = None


def _deq(q, s):
    """fp4x2 bytes [rows, K/2] + e8m0 [rows, K/32] -> fp32 [rows, K]."""
    global _LUT
    if _LUT is None:
        _LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], device=q.device)
    v = torch.stack([_LUT[(q & 15).long()], _LUT[(q >> 4).long()]], -1).reshape(q.shape[0], -1)
    return v * torch.pow(2.0, s.float() - 127).repeat_interleave(32, -1)


def make(M, N, K, batch=1, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randint(0, 256, (batch, M, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    w = torch.randint(0, 256, (batch, N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
    xs = torch.randint(124, 131, (batch, M, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    ws = torch.randint(124, 131, (batch, N, K // 32), dtype=torch.uint8, device="cuda", generator=g)
    fp4, e8 = aiter.dtypes.fp4x2, aiter.dtypes.fp8_e8m0
    wsh = torch.stack([shuffle_weight(w[b].view(fp4), layout=(16, 16)) for b in range(batch)])
    xss = torch.stack([e8m0_shuffle(xs[b].view(e8)) for b in range(batch)])
    wss = torch.stack([e8m0_shuffle(ws[b].view(e8)) for b in range(batch)])
    return x, w, xs, ws, x.view(fp4), wsh, xss, wss


def ref(x, w, xs, ws):
    return torch.stack([_deq(x[b], xs[b]) @ _deq(w[b], ws[b]).T for b in range(x.shape[0])])


def check(out, r, name):
    o = out.float()
    err = ((o - r).norm() / r.norm()).item()
    ok = torch.isfinite(o).all().item() and err < 1e-2
    print(f"  {name:28s} rel_err {err:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def bench(f, warm=200, it=100):
    for _ in range(warm):
        f()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(3):
        e0.record()
        for _ in range(it):
            f()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / it)
    return sorted(ts)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true")
    a = ap.parse_args()
    shapes = [(256, 256, 512, 1), (1024, 2048, 1024, 2), (300, 400, 768, 2), (1, 128, 256, 1), (100, 1280, 7168, 1),
              (1000, 4112, 256, 1), (16, 4096, 4096, 1)]
    if a.perf:
        shapes = [(8192, 8192, 8192, 1), (2048, 4096, 8192, 1), (1024, 4096, 8192, 1)]
    ok = True
    for M, N, K, b in shapes:
        x, w, xs, ws, xq, wsh, xss, wss = make(M, N, K, b)
        print(f"M={M} N={N} K={K} batch={b}")
        r = ref(x, w, xs, ws) if M * N * K * b <= 2 ** 34 else None
        out = gemm_a4w4_opus(xq, wsh, xss, wss)
        if r is not None:
            ok &= check(out, r, "gemm_a4w4_opus (dispatch)")
        for kid, t in KIDS.items():
            o = opus_gemm_a4w4_tune(xq, wsh, xss, wss, kernelId=kid)
            if r is not None:
                ok &= check(o, r, f"kid {kid} ({t})")
            if a.perf:
                us = bench(lambda: opus_gemm_a4w4_tune(xq, wsh, xss, wss, Y=o, kernelId=kid))
                print(f"    {t}: {us:7.1f} us  {2 * M * N * K * b / us / 1e9:.0f} TFLOPS")
        if b == 1 and M % 32 == 0:
            try:
                from aiter.ops.gemm_op_a4w4 import gemm_a4w4_asm
                y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
                fa = lambda: gemm_a4w4_asm(xq[0], wsh[0], xss[0], wss[0], y, "", None, 1.0, 0.0, True)
                fa()
                if r is not None:
                    check(y.unsqueeze(0), r, "aiter gemm_a4w4 asm")
                if a.perf:
                    us = bench(fa)
                    print(f"    asm: {us:7.1f} us  {2 * M * N * K / us / 1e9:.0f} TFLOPS")
            except Exception as e:  # noqa: BLE001
                print(f"  aiter asm unavailable: {type(e).__name__}: {str(e)[:120]}")
        if a.perf:
            o = gemm_a4w4_opus(xq, wsh, xss, wss)
            us = bench(lambda: gemm_a4w4_opus(xq, wsh, xss, wss, Y=o))
            print(f"    dispatch: {us:7.1f} us")
    # Unsupported shapes must be rejected (Python check and, bypassing it, the C++ AITER_CHECK), not silently miscomputed.
    if not a.perf:
        from aiter.ops.opus import gemm_op_a4w4 as op
        fp4, e8 = aiter.dtypes.fp4x2, aiter.dtypes.fp8_e8m0
        for M, N, K, why in ((64, 100, 256, "N % 16"), (64, 128, 320, "K % 256")):
            x = torch.zeros((1, M, K // 2), dtype=torch.uint8, device="cuda").view(fp4)
            w = torch.zeros((1, N, K // 2), dtype=torch.uint8, device="cuda").view(fp4)
            cols = (K // 32 + 7) // 8 * 8
            xs = torch.full((256, cols), 127, dtype=torch.uint8, device="cuda").view(e8)
            ws = torch.full(((N + 255) // 256 * 256, cols), 127, dtype=torch.uint8, device="cuda").view(e8)
            calls = (("python", lambda: gemm_a4w4_opus(x, w, xs, ws)),
                     ("c++", lambda: op._opus_gemm_a4w4_tune_raw(
                         x, w, torch.empty((1, M, N), dtype=torch.bfloat16, device="cuda"), xs, ws, 30000)))
            for label, call in calls:
                try:
                    call()
                    torch.cuda.synchronize()
                    print(f"  M={M} N={N} K={K} ({why}) {label}: NOT rejected -> FAIL")
                    ok = False
                except Exception as e:  # noqa: BLE001
                    print(f"  M={M} N={N} K={K} ({why}) {label}: rejected OK ({str(e).splitlines()[0][:90]})")
    print("ALL OK" if ok else "SOME FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
