# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""OPUS a8w4 (gfx950): MXFP8 A x MXFP4 B GEMM, 1x32 e8m0_shuffle scales, bpreshuffle B.

Runs every kid in NT / NN / TT / TN (split-K kids at S = 2 / 4) through
opus_gemm(family="a8w4", layout="bpreshuffle") against an fp32 reference
decoded independently of the kernels, outputs of 4 GiB and more, and the
inputs the launcher must reject.

Layouts (first letter A, second B): T = A stored [K, M] (passed as its [M, K]
view, M % 16); N = B stored [K, N/2] (n pairs packed, N % 32; passed as the
[N/2, K] view of its shuffle_weight). x_scale / w_scale are the e8m0_shuffle of
[M, K/32] / [N, K/32] in every layout.
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import e8m0_shuffle

# Ragged M / N tails, an N below every tile width, and a tile-aligned shape (the
# only one the "nob" kid runs); M % 16 and N % 32 so every layout is legal.
DEFAULT_MNK = [
    (304, 416, 768),
    (1008, 4128, 2048),
    (16, 128, 256),
    (2048, 1536, 7168),
]
# A 4 GiB C: a 32-bit buffer range over all of C wraps here, so these need the
# kernel's per-tile 64-bit C base (and fault or drop rows without it).
DEFAULT_LARGE_MNK = [(32768, 65536, 256)]
LAYOUTS = ["NT", "NN", "TT", "TN"]
_FP4 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def registry():
    from csrc.opus_gemm.opus_gemm_common import get_registry

    return get_registry("a8w4", get_gfx(), get_cu_num())


def quantize(m, n, k):
    """MXFP8 A [M, K] and MXFP4 B [N, K/2] (k pairs packed) with their raw e8m0 scales."""
    x = torch.randn(m, k, dtype=dtypes.bf16)
    w = torch.randn(n, k, dtype=dtypes.bf16)
    xq, x_scale = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    wq, w_scale = aiter.get_triton_quant(aiter.QuantType.per_1x32)(w, shuffle=False)
    return xq, x_scale, wq.view(torch.uint8), w_scale.view(dtypes.fp8_e8m0)


def operands(xq, x_scale, wq, w_scale, layout):
    """The operands as the layout stores them, and the shuffled scales."""
    a = xq if layout[0] == "N" else xq.T.contiguous().T
    if layout[1] == "T":
        b = shuffle_weight(wq.view(dtypes.fp4x2), layout=(16, 16))
    else:  # B stored [K, N/2]: nibble (k, n) of [N, K/2] moved to byte (k, n / 2)
        n, kb = wq.shape
        codes = torch.stack([wq & 15, wq >> 4], -1).view(n, 2 * kb).T
        bt = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        b = shuffle_weight(bt.view(dtypes.fp4x2), layout=(16, 16)).T
    return a, b, e8m0_shuffle(x_scale), e8m0_shuffle(w_scale)


def reference(xq, x_scale, wq, w_scale, rows):
    """fp32 C of A rows ``rows``: fp8 / fp4 codes times 2^(e8m0 - 127) per 32 K."""
    lut = torch.tensor(_FP4, dtype=torch.float32)
    a = xq[rows].float() * torch.exp2(
        x_scale[rows].view(torch.uint8).float() - 127
    ).repeat_interleave(32, 1)
    n, kb = wq.shape
    b = torch.stack([lut[(wq & 15).long()], lut[(wq >> 4).long()]], -1).view(n, 2 * kb)
    b = b * torch.exp2(w_scale.view(torch.uint8).float() - 127).repeat_interleave(32, 1)
    return a @ b.T


def run_opus(a, b, y, x_scale, w_scale, kid, split_k):
    from aiter.ops.opus import opus_gemm

    return opus_gemm(
        a,
        b,
        y,
        kid=kid,
        family="a8w4",
        layout="bpreshuffle",
        x_scale=x_scale,
        w_scale=w_scale,
        split_k=split_k,
    )


def opus_err(xq, x_scale, wq, w_scale, y):
    """Mismatch fraction against the fp32 reference, row block by row block
    (a 4 GiB output's reference does not fit at once); the worst block counts."""
    m, n = y.shape
    step = max(1, (1 << 27) // n)
    return max(
        checkAllclose(
            reference(xq, x_scale, wq, w_scale, slice(r, r + step)),
            y[r : r + step].float(),
            printLog=False,
        )
        for r in range(0, m, step)
    )


@benchmark()
def test_opus(m, n, k, layout, kid, split_k):
    xq, x_scale, wq, w_scale = quantize(m, n, k)
    a, b, xs, ws = operands(xq, x_scale, wq, w_scale, layout)
    # NaN sentinel: an element the kernel leaves unwritten fails the comparison.
    y = torch.full((m, n), float("nan"), dtype=dtypes.bf16)
    run_opus(a, b, y, xs, ws, kid, split_k)
    err = opus_err(xq, x_scale, wq, w_scale, y)
    # No rotated input copies for a multi-GiB output (each would be a full C).
    rotate = 1 if 2 * m * n >= 1 << 31 else 0
    _, us = run_perftest(
        run_opus, a, b, y, xs, ws, kid, split_k, num_rotate_args=rotate
    )
    assert (
        err == 0
    ), f"OPUS a8w4 kid {kid} {layout} M={m} N={n} K={k} S={split_k}: err {err}"
    return {
        "gfx": get_gfx(),
        "kernel": registry()[kid].name,
        "us": round(us, 2),
        "TFLOPS": round(2 * m * n * k / us / 1e6, 1),
        "TB/s": round((m * k + n * k // 2 + m * n * 2) / us / 1e6, 2),
        "err": err,
    }


def check_rejects():
    """Inputs the launcher must refuse (a wrong result here is silent)."""
    kids = registry()
    sk = next(kid for kid, inst in kids.items() if inst.co_variant == "sk")
    nob = next(kid for kid, inst in kids.items() if inst.co_variant == "nob")
    np_kid = next(kid for kid, inst in kids.items() if inst.co_variant == "np")
    xq, x_scale, wq, w_scale = quantize(304, 416, 768)
    a, b, xs, ws = operands(xq, x_scale, wq, w_scale, "NT")
    padded = torch.empty(304, 768 + 64, dtype=xq.dtype)[:, :768]
    padded.copy_(xq)
    # A stored [K, M] needs M % 16; 312 % 16 = 8.
    xq_t, x_scale_t, wq_t, w_scale_t = quantize(312, 416, 768)
    a_t, b_t, xs_t, ws_t = operands(xq_t, x_scale_t, wq_t, w_scale_t, "TT")
    cases = {
        "non-dense XQ strides": (padded, b, xs, ws, np_kid, 0),
        "A stored [K, M] with M % 16": (a_t, b_t, xs_t, ws_t, np_kid, 0),
        "nob kid on a ragged M": (a, b, xs, ws, nob, 0),
        "split_k on a non-split-K kid": (a, b, xs, ws, np_kid, 2),
        "K not a multiple of split_k * 256": (a, b, xs, ws, sk, 8),
        "unshuffled x_scale (K/32 columns)": (a, b, x_scale, ws, np_kid, 0),
    }
    rows = []
    for case, (aa, bb, sa, sb, kid, split_k) in cases.items():
        y = torch.empty(aa.shape[0], 416, dtype=dtypes.bf16)
        try:
            run_opus(aa, bb, y, sa, sb, kid, split_k)
            torch.cuda.synchronize()
            rejected = False
        except (RuntimeError, ValueError):
            rejected = True
        assert rejected, f"OPUS a8w4 accepted: {case}"
        rows.append({"case": case, "kid": kid, "rejected": rejected})
    from aiter.ops.opus.gemm_op_a8w4 import a8w4_kid_runs

    over = not a8w4_kid_runs(kids[sk], 8192, 65536, 1536, 2)
    assert over, "split-K workspace over 4 GiB accepted"
    rows.append({"case": "split-K workspace > 4 GiB", "kid": sk, "rejected": over})
    return rows


def cases(mnks, layouts):
    """(m, n, k, layout, kid, split_k) for every kid that runs the shape."""
    from aiter.ops.opus.gemm_op_a8w4 import a8w4_kid_runs

    for (m, n, k), layout, (kid, inst) in itertools.product(
        mnks, layouts, sorted(registry().items())
    ):
        for split_k in (2, 4) if inst.co_variant == "sk" else (0,):
            if a8w4_kid_runs(inst, m, n, k, split_k, layout):
                yield m, n, k, layout, kid, split_k


def summarize(title, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", title, df.to_markdown(index=False))


def main():
    torch.set_default_device("cuda")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_MNK,
        help="kid sweep shapes (M % 16, N % 32 keep every layout legal)",
    )
    parser.add_argument(
        "--large_mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=DEFAULT_LARGE_MNK,
        help="shapes with a C of 4 GiB or more",
    )
    parser.add_argument(
        "-l",
        "--layout",
        nargs="*",
        default=LAYOUTS,
        choices=LAYOUTS,
        help="operand layouts (first letter A, second B; T = A stored [K, M], N = B stored [K, N/2])",
    )
    args = parser.parse_args()

    if get_gfx() != "gfx950":
        aiter.logger.warning("OPUS a8w4 is gfx950 only; skipping on %s", get_gfx())
        return
    torch.manual_seed(0)
    summarize("OPUS a8w4", [test_opus(*c) for c in cases(args.mnk, args.layout)])
    large = [
        test_opus(m, n, k, "NT", kid, split_k)
        for m, n, k in args.large_mnk
        for kid, split_k in ((0, 0), (1, 0), (7, 0))
    ]
    summarize("OPUS a8w4 4 GiB C", large)
    summarize("OPUS a8w4 rejects", check_rejects())


if __name__ == "__main__":
    main()
