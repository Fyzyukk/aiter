# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness / performance of the opus gfx950 MXFP4 GEMM (a4w4_mxfp4_nt, bpreshuffle=True, and a4w4_mxfp4_nt_noshuffle,
bpreshuffle=False) vs a torch reference and aiter gemm_a4w4 asm.

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

import os
_T = ("256x256", "256x128", "128x256", "128x128", "192x256", "256x192")
KIDS = {**{30000 + i: t for i, t in enumerate(_T)}, **{30010 + i: t + " np" for i, t in enumerate(_T)}, 30030: "256x256 nob"}
SK_KIDS = {30020 + i: t + " sk" for i, t in enumerate(_T) if i}
NS_KIDS = {**{30100 + i: t + " ns" for i, t in enumerate(_T)}, **{30110 + i: t + " ns np" for i, t in enumerate(_T)}, 30130: "256x256 ns nob"}
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
    # shuffle_weight(16, 16) needs N % 16 == 0: other N only run the bpreshuffle=False path
    wsh = torch.stack([shuffle_weight(w[b].view(fp4), layout=(16, 16)) for b in range(batch)]) if N % 16 == 0 else None
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
              (1000, 4112, 256, 1), (16, 4096, 4096, 1), (50, 136, 768, 2)]
    if a.perf:
        shapes = [(8192, 8192, 8192, 1), (2048, 4096, 8192, 1), (1024, 4096, 8192, 1)]
    ok = True
    for M, N, K, b in shapes:
        x, w, xs, ws, xq, wsh, xss, wss = make(M, N, K, b)
        print(f"M={M} N={N} K={K} batch={b}")
        r = ref(x, w, xs, ws) if M * N * K * b <= 2 ** 34 else None
        shuffled = wsh is not None
        if shuffled:
            out = gemm_a4w4_opus(xq, wsh, xss, wss)
            if r is not None:
                ok &= check(out, r, "gemm_a4w4_opus (dispatch)")
        for kid, t in (KIDS.items() if shuffled else ()):
            if kid == 30030 and (M % 256 or N % 256):
                continue
            o = opus_gemm_a4w4_tune(xq, wsh, xss, wss, kernelId=kid)
            if r is not None:
                ok &= check(o, r, f"kid {kid} ({t})")
            if a.perf:
                us = bench(lambda: opus_gemm_a4w4_tune(xq, wsh, xss, wss, Y=o, kernelId=kid))
                print(f"    {t}: {us:7.1f} us  {2 * M * N * K * b / us / 1e9:.0f} TFLOPS")
        for kid, t in (SK_KIDS.items() if shuffled else ()):
            for S in (1, 2, 4, 8):
                if K % (S * 256):
                    continue
                os.environ["OPUS_MXFP4_SPLITK"] = str(S)
                o = opus_gemm_a4w4_tune(xq, wsh, xss, wss, kernelId=kid)
                if r is not None:
                    ok &= check(o, r, f"kid {kid} ({t} S={S})")
                del os.environ["OPUS_MXFP4_SPLITK"]
        # bpreshuffle=False: plain row-major WQ and [rows, K/32] scales (no padding).
        e8 = aiter.dtypes.fp8_e8m0
        wq, xsr, wsr = w.view(aiter.dtypes.fp4x2), xs.view(e8), ws.view(e8)
        out = gemm_a4w4_opus(xq, wq, xsr, wsr, bpreshuffle=False)
        if r is not None:
            ok &= check(out, r, "gemm_a4w4_opus bpreshuffle=False")
        xs2, ws2 = xsr.reshape(-1, K // 32), wsr.reshape(-1, K // 32)   # 2D [batch * rows, K/32] form
        out = gemm_a4w4_opus(xq, wq, xs2, ws2, bpreshuffle=False)
        if r is not None:
            ok &= check(out, r, "  (2D scales)")
        for kid, t in NS_KIDS.items():
            if kid == 30130 and (M % 256 or N % 256):
                continue
            o = opus_gemm_a4w4_tune(xq, wq, xsr, wsr, kernelId=kid)
            if r is not None:
                ok &= check(o, r, f"kid {kid} ({t})")
            if a.perf:
                us = bench(lambda: opus_gemm_a4w4_tune(xq, wq, xsr, wsr, Y=o, kernelId=kid))
                print(f"    {t}: {us:7.1f} us  {2 * M * N * K * b / us / 1e9:.0f} TFLOPS")
        if b == 1 and M % 32 == 0 and shuffled:
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
        if a.perf and shuffled:
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
        # bpreshuffle=False: N % 8, K % 256, scales must be row-major [rows, K/32] (an e8m0_shuffle'd one has K/32 padded to 8).
        for M, N, K, why, sfcols in ((64, 100, 256, "ns N % 8", None), (64, 128, 320, "ns K % 256", None),
                                     (64, 128, 256, "ns scale cols != K/32", 16)):
            x = torch.zeros((1, M, K // 2), dtype=torch.uint8, device="cuda").view(fp4)
            w = torch.zeros((1, N, K // 2), dtype=torch.uint8, device="cuda").view(fp4)
            c = sfcols or K // 32
            xs = torch.full((M, c), 127, dtype=torch.uint8, device="cuda").view(e8)
            ws = torch.full((N, c), 127, dtype=torch.uint8, device="cuda").view(e8)
            calls = (("python", lambda: gemm_a4w4_opus(x, w, xs, ws, bpreshuffle=False)),
                     ("c++", lambda: op._opus_gemm_a4w4_raw(
                         x, w, torch.empty((1, M, N), dtype=torch.bfloat16, device="cuda"), xs, ws, False)),
                     ("c++ tune", lambda: op._opus_gemm_a4w4_tune_raw(
                         x, w, torch.empty((1, M, N), dtype=torch.bfloat16, device="cuda"), xs, ws, 30100)))
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
