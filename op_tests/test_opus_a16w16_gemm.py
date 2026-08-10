# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""End-to-end regression of gemm_a16w16_opus vs torch.bmm; prints TFLOPs.

Usage:
    python3 op_tests/test_opus_a16w16_gemm.py [-m M -n N -k K -b B]
    python3 op_tests/test_opus_a16w16_gemm.py --csv_file <shape_csv>
"""

import argparse
import sys

import pytest
import torch

# Skip on unsupported arch via the same probe opus uses at import time.
from aiter.ops.opus._arch import _detect_arch

_arch_ok, _detected_gfx = _detect_arch({"gfx950", "gfx942", "gfx1250"})

from aiter.ops.opus import gemm_a16w16_opus
from aiter.test_common import checkAllclose, run_perftest


def _torch_ref(A: torch.Tensor, B: torch.Tensor, out_dtype):
    # A: [batch, M, K], B: [N, K] or [batch, N, K] -> bmm.
    # run_torch computes in fp32 then casts to match the opus path.
    if B.dim() == 2:
        return torch.einsum("bmk,nk->bmn", A.float(), B.float()).to(out_dtype)
    return torch.bmm(A.float(), B.float().transpose(-1, -2)).to(out_dtype)


def _make_b(batch: int, N: int, K: int) -> torch.Tensor:
    """Build a B that gemm_a16w16_opus accepts for both batch=1 and batch>1.

    The wrapper rejects 2D B + batch>1 because the opus launcher hardcodes
    stride_b_batch == N*K (a broadcast view would silently fault). For the
    common "shared weight across batch" case, materialize an explicit
    `[batch, N, K]` tensor via the contiguous broadcast pattern.
    """
    B2D = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    if batch == 1:
        return B2D
    return B2D.unsqueeze(0).expand(batch, -1, -1).contiguous()


def run_a16w16_case(batch: int, M: int, N: int, K: int, out_dtype=torch.bfloat16):
    # gemm_a16w16_opus accepts either 2D or 3D A; test 3D to exercise the
    # batched reshape path. B is 2D when batch==1, 3D contiguous otherwise.
    A = torch.randn(batch, M, K, device="cuda", dtype=torch.bfloat16)
    B = _make_b(batch, N, K)

    ref = _torch_ref(A, B, out_dtype)

    Y, us = run_perftest(
        gemm_a16w16_opus,
        A,
        B,
        None,
        out_dtype,
    )

    err = checkAllclose(
        Y,
        ref,
        msg=f"a16w16 b={batch} m={M} n={N} k={K}",
        rtol=0.1,
        atol=0.5,
    )
    flops = 2.0 * batch * M * N * K
    tflops = flops / us / 1e6
    print(
        f"[a16w16] batch={batch} M={M} N={N} K={K} dtype={out_dtype} "
        f"| {us:.1f}us | {tflops:.2f} TFLOPs | err={err}"
    )
    return err


def load_shapes_from_csv(csv_path):
    import pandas as pd

    df = pd.read_csv(csv_path)
    shapes = list(zip(df["M"].astype(int), df["N"].astype(int), df["K"].astype(int)))
    return list(dict.fromkeys(shapes))


def run_a16w16_csv_sweep(csv_path: str, batch: int = 1):
    shapes = load_shapes_from_csv(csv_path)
    print(f"\n{'=' * 80}")
    print(f"a16w16 sweep from {csv_path}: {len(shapes)} unique shapes, batch={batch}")
    print("=" * 80)
    passed = failed = 0
    for M, N, K in shapes:
        tag = f"a16w16 b={batch} M={M} N={N} K={K}"
        try:
            A = torch.randn(batch, M, K, device="cuda", dtype=torch.bfloat16)
            B = _make_b(batch, N, K)
            ref = _torch_ref(A, B, torch.bfloat16)
            Y, us = run_perftest(
                gemm_a16w16_opus,
                A,
                B,
                None,
                torch.bfloat16,
            )
            err = checkAllclose(Y, ref, msg=tag, rtol=0.1, atol=0.5)
            tflops = 2.0 * batch * M * N * K / us / 1e6
            print(f"[PASS] {tag} | {us:.1f}us | {tflops:.2f} TFLOPs | err={err}")
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {tag} | {type(e).__name__}: {e}")
            failed += 1
    print(f"\nSummary: {passed} passed, {failed} failed out of {len(shapes)}")
    return failed == 0


def _runtime_arch() -> str | None:
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()


def _assert_matches_golden(actual, A, B, bias=None):
    golden = A.float() @ B.float().transpose(-1, -2)
    if bias is not None:
        golden = golden + bias.float()
    # BF16 output has one final rounding; fp32 output is normally much tighter.
    atol = 0.5 if actual.dtype == torch.bfloat16 else 0.05
    rtol = 0.03 if actual.dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(actual.float(), golden, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    ("arch", "kid", "M", "N", "K", "split_k", "out_dtype"),
    [
        ("gfx950", 200, 64, 64, 512, 2, torch.bfloat16),
        ("gfx950", 200, 64, 64, 512, 2, torch.float32),
        ("gfx942", 10200, 128, 128, 512, 2, torch.float32),
        ("gfx942", 10210, 128, 128, 512, 2, torch.bfloat16),
        ("gfx1250", 20000, 16, 32, 512, 2, torch.bfloat16),
        ("gfx1250", 20000, 16, 32, 512, 2, torch.float32),
    ],
)
def test_split_k_matches_torch_golden(arch, kid, M, N, K, split_k, out_dtype):
    if _runtime_arch() != arch:
        pytest.skip(f"requires {arch} hardware")
    torch.manual_seed(8192 + kid)
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    actual = gemm_a16w16_opus(
        A,
        B,
        dtype=out_dtype,
        kernelId=kid,
        splitK=split_k,
    )
    torch.cuda.synchronize()
    _assert_matches_golden(actual, A, B)


def test_gfx950_bias_dtype_rules_and_numerics():
    if _runtime_arch() != "gfx950":
        pytest.skip("requires gfx950 hardware")
    A = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((64, 512), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((64,), device="cuda", dtype=torch.bfloat16)
    actual = gemm_a16w16_opus(
        A,
        B,
        bias=bias,
        dtype=torch.bfloat16,
        kernelId=200,
        splitK=2,
    )
    torch.cuda.synchronize()
    _assert_matches_golden(actual, A, B, bias)

    with pytest.raises(RuntimeError, match="bias dtype must match Y dtype"):
        gemm_a16w16_opus(
            A,
            B,
            bias=bias.float(),
            dtype=torch.bfloat16,
            kernelId=200,
            splitK=2,
        )


def test_gfx942_bias_keeps_framework_fallback_and_raw_rejection():
    if _runtime_arch() != "gfx942":
        pytest.skip("requires gfx942 hardware")
    A = torch.randn((128, 4096), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((256, 4096), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((256,), device="cuda", dtype=torch.float32)

    # Automatic selection keeps the established framework fallback semantics.
    actual = gemm_a16w16_opus(A, B, bias=bias, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    _assert_matches_golden(actual, A, B, bias)

    # An explicit workspace kid remains a strict error instead of silently
    # changing the requested launcher.
    with pytest.raises(ValueError, match="rejects bias on split-K kernels"):
        gemm_a16w16_opus(
            A,
            B,
            bias=bias,
            dtype=torch.bfloat16,
            kernelId=10201,
            splitK=2,
        )


def test_gfx1250_bf16_output_accepts_fp32_bias():
    if _runtime_arch() != "gfx1250":
        pytest.skip("requires gfx1250 hardware")
    A = torch.randn((16, 512), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((32, 512), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((32,), device="cuda", dtype=torch.float32)
    actual = gemm_a16w16_opus(
        A,
        B,
        bias=bias,
        dtype=torch.bfloat16,
        kernelId=20000,
        splitK=2,
    )
    torch.cuda.synchronize()
    _assert_matches_golden(actual, A, B, bias)


if __name__ == "__main__":
    if not _arch_ok:
        print(
            "[skip] test_opus_a16w16_gemm requires "
            f"gfx950/gfx942/gfx1250 (detected {_detected_gfx!r})"
        )
        sys.exit(0)
    parser = argparse.ArgumentParser(
        description="End-to-end test for aiter.ops.opus.gemm_a16w16_opus"
    )
    parser.add_argument("-m", type=int, default=256)
    parser.add_argument("-n", type=int, default=512)
    parser.add_argument("-k", type=int, default=256)
    parser.add_argument("-b", "--batch", type=int, default=8)
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Output dtype (default: bf16)",
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        default=None,
        metavar="CSV",
        help=(
            "Optional CSV with M,N,K columns. When given, skips the "
            "single-shape test and runs a full sweep instead."
        ),
    )
    args = parser.parse_args()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.csv_file is not None:
        run_a16w16_csv_sweep(args.csv_file, batch=args.batch)
    else:
        # Clamp K>=128 so every kid the heuristic picks has K>=B_K (smallest is 128).
        k_eff = max(args.k, 128)
        run_a16w16_case(args.batch, args.m, args.n, k_eff, out_dtype=out_dtype)
