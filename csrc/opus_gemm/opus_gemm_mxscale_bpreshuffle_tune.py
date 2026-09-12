# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gfx950 OPUS FP8 B-preshuffle GEMM with native compact E8M0 scales."""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch

from aiter import logger
from aiter.ops.opus import opus_gemm
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner
from csrc.opus_gemm.opus_gemm_common import (
    a8w8_mxscale_gemm_bpreshuffle_kernels_list,
    canonical_output_dtype,
)

_TAG = "a8w8_mxscale_gemm_bpreshuffle"
_DEFAULT_HIP_CLANG_PATH = "/root/toolchains/rocm-llvm23-46fcb339-build/bin"
_BENCH_KEYS = ("x", "w", "out", "x_scale", "w_scale")
_REF_KEYS = ("x", "w_reference", "x_scale", "w_scale")


def _ensure_kids_compiled(candidate_kids):
    """Reuse the canonical OPUS subset builder with the patched clang23."""
    candidate_kids = frozenset(int(kid) for kid in candidate_kids)
    if not candidate_kids:
        return False

    compiler_path = os.environ.get("OPUS_HIP_CLANG_PATH", _DEFAULT_HIP_CLANG_PATH)
    if not Path(compiler_path).is_dir():
        raise FileNotFoundError(
            "The 4-wave MXFP8 kernel requires the patched clang23 toolchain; "
            f"directory not found: {compiler_path}. Set OPUS_HIP_CLANG_PATH "
            "to its bin directory."
        )

    # opus_gemm_tune.py is also a directly executable script and therefore
    # uses sibling absolute imports. Load it lazily only when tuning/replaying,
    # after making that sibling directory importable.
    opus_dir = str(Path(__file__).resolve().parent)
    added_path = opus_dir not in sys.path
    if added_path:
        sys.path.insert(0, opus_dir)
    try:
        from opus_gemm_tune import _ensure_kids_compiled as ensure_existing_opus_kids

        previous_hip_clang_path = os.environ.get("HIP_CLANG_PATH")
        os.environ["HIP_CLANG_PATH"] = compiler_path
        try:
            return ensure_existing_opus_kids(candidate_kids)
        finally:
            if previous_hip_clang_path is None:
                os.environ.pop("HIP_CLANG_PATH", None)
            else:
                os.environ["HIP_CLANG_PATH"] = previous_hip_clang_path
    finally:
        if added_path:
            sys.path.remove(opus_dir)


def candidate_kids_for_shape(gfx, m, n, k, outdtype="bf16"):
    """Use registry tiling and byte limits before allocating tuning inputs."""
    if min(m, n, k) <= 0:
        return []
    canonical_out = canonical_output_dtype(outdtype)
    out_bytes = {"bf16_t": 2, "fp32_t": 4}.get(canonical_out)
    if out_bytes is None:
        return []
    candidates = []
    for kid, instance in sorted(a8w8_mxscale_gemm_bpreshuffle_kernels_list.items()):
        if (
            gfx != (instance.arch_prefix or "gfx950")
            or instance.kernel_tag != _TAG
            or canonical_out not in instance.output_dtypes
        ):
            continue
        if m % instance.m_align or n % instance.B_N or k % instance.B_K:
            continue
        if max(m * k, n * k, m * n * out_bytes) > instance.max_tensor_bytes:
            continue
        candidates.append(kid)
    return candidates


def generate_data(m, n, k, kid, *, device):
    """Construct paired FP8/E8M0 operands; preprocessing is outside timing."""
    instance = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((m, k), device=device, generator=generator).to(torch.float8_e4m3fn)
    w = torch.randn((n, k), device=device, generator=generator).to(x.dtype)
    kg = k // instance.GROUP_K
    # Native E8M0 exponent bytes: nonuniform finite powers of two, not rounded
    # FP32 scales paired with an unrelated prequantized activation.
    x_scale = (
        torch.randint(
            123, 130, (kg, m), device=device, dtype=torch.uint8, generator=generator
        )
        .view(torch.float8_e8m0fnu)
        .T
    )
    w_scale = torch.randint(
        123,
        130,
        (n // instance.GROUP_N, kg),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    ).view(torch.float8_e8m0fnu)
    return {
        "x": x,
        "w": shuffle_weight(w, layout=(16, 16)),
        "w_reference": w,
        "out": torch.empty((m, n), device=device, dtype=torch.bfloat16),
        "x_scale": x_scale,
        "w_scale": w_scale,
    }


def run_torch(x, w, x_scale, w_scale, *, with_bounds=False):
    """Independent FP32 reference and optional absolute-product magnitudes."""
    a = x.float() * x_scale.float().repeat_interleave(x.shape[1] // x_scale.shape[1], 1)
    b = w.float() * w_scale.float().repeat_interleave(
        w.shape[0] // w_scale.shape[0], 0
    ).repeat_interleave(w.shape[1] // w_scale.shape[1], 1)
    ref = a @ b.T
    if with_bounds:
        # The scaled MFMA's accumulation error depends on the summed product
        # magnitudes, including when positive/negative terms cancel to zero.
        # Stack the two reference planes so mp_tuner passes them to one
        # comparison callback for the single output Tensor.
        return torch.stack((ref, a.abs() @ b.abs().T))
    return ref


def run_bench(x, w, out, x_scale, w_scale, kid):
    opus_gemm(
        x,
        w,
        out,
        kid=kid,
        layout="bpreshuffle",
        x_scale=x_scale,
        w_scale=w_scale,
    )
    return out


def compare_outputs(ref, out, **kwargs):
    """Check every value against the source kernel's accumulation contract.

    The prototype's valid_vector uses 1e-4 + 5e-5*sum(abs(A_i*B_i))
    for FP32 accumulation, then rounds both interval endpoints to BF16 for
    BF16 output. This handles cancellation without allowing a nonzero
    fraction of arbitrary outliers. Reference construction is never timed.
    """
    expected, magnitude = ref.unbind(0)
    if expected.shape != out.shape:
        raise ValueError("MXFP8 reference and output shapes must match")
    if not torch.isfinite(ref).all() or not torch.isfinite(out).all():
        return 1.0
    error = 1e-4 + 5e-5 * magnitude
    lower = (expected - error).to(out.dtype).float()
    upper = (expected + error).to(out.dtype).float()
    if not torch.isfinite(lower).all() or not torch.isfinite(upper).all():
        return 1.0
    actual = out.float()
    mismatches = int(((actual < lower) | (actual > upper)).sum().item())
    if kwargs.get("printLog", True):
        logger.info(
            f"{kwargs.get('msg', '')}MXFP8 accumulation bounds: "
            f"{mismatches}/{out.numel()} elements outside the allowed interval"
        )
    # mp_tuner saves four decimals; even one mismatch must remain nonzero.
    return math.ceil(mismatches / out.numel() * 10000) / 10000


class OpusMxscaleBpreshuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": "/tmp/opus_mxscale_bpreshuffle_tuned.csv",
        "errRatio": 0.0,
    }

    def __init__(self):
        super().__init__(
            "opus_mxscale_bpreshuffle",
            # Input/output/scale dtypes are fixed by this specialized tuner.
            # Keep them as validated internal defaults, not serialized shape
            # keys, so -o matches the production tuned-config CSV schema.
            key=["gfx", "cu_num", "M", "N", "K"],
            resultList=[
                "libtype",
                "kernelId",
                "splitK",
                "us",
                "kernelName",
                "tflops",
                "bw",
                "errRatio",
            ],
            description="Tune gfx950 FP8 B-preshuffle GEMM with E8M0 A 1x128 / B 128x128 scales and BF16 output. Measures GEMM only.",
        )

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--input_file", dest="untune_file", default=argparse.SUPPRESS
        )
        self.parser.add_argument(
            "--tuned_file", dest="tune_file", default=argparse.SUPPRESS
        )

    def _normalize_rows(self, df, *, saved=False):
        df = df.copy()
        missing = {"M", "N", "K"}.difference(df.columns)
        if missing:
            raise ValueError(f"Shape CSV is missing columns: {sorted(missing)}")
        defaults = {
            "gfx": self.get_gfx(),
            "cu_num": self.get_cu_num(),
            "dtype": str(torch.float8_e4m3fn),
            "outdtype": str(torch.bfloat16),
            "scale_dtype": "e8m0",
        }
        if saved:
            missing = {
                "gfx",
                "cu_num",
                "libtype",
                "kernelId",
                "splitK",
            }.difference(df.columns)
            if missing:
                raise ValueError(
                    f"Saved MXFP8 CSV is missing columns: {sorted(missing)}"
                )
        for column, default in defaults.items():
            if column not in df:
                df[column] = default
        for column in (
            "cu_num",
            "M",
            "N",
            "K",
            *(("kernelId", "splitK") if saved else ()),
        ):
            value = pd.to_numeric(df[column], errors="raise")
            minimum = 0 if column in ("kernelId", "splitK") else 1
            if (value.isna() | (value < minimum) | (value % 1 != 0)).any():
                raise ValueError(f"{column} must contain integers >= {minimum}")
            df[column] = value.astype("int64")
        if (
            not df["dtype"]
            .isin(["fp8", "float8_e4m3fn", str(torch.float8_e4m3fn)])
            .all()
        ):
            raise ValueError("MXFP8 bpreshuffle tune requires FP8 E4M3FN inputs")
        if not df["outdtype"].map(canonical_output_dtype).eq("bf16_t").all():
            raise ValueError("MXFP8 bpreshuffle tune requires BF16 output")
        if (
            not df["scale_dtype"]
            .isin(["e8m0", "float8_e8m0fnu", str(torch.float8_e8m0fnu)])
            .all()
        ):
            raise ValueError("MXFP8 bpreshuffle tune requires E8M0 scales")
        if saved and (
            not df["libtype"].eq("opus").all() or not df["splitK"].eq(0).all()
        ):
            raise ValueError("Saved MXFP8 rows require libtype=opus and splitK=0")
        df["dtype"], df["outdtype"], df["scale_dtype"] = (
            str(torch.float8_e4m3fn),
            str(torch.bfloat16),
            "e8m0",
        )
        return df

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        df = super().get_tuned_gemm_list(tuned_gemm_file, columns)
        if df.empty:
            return df
        # Accept both legacy files carrying redundant dtype columns and the
        # compact production schema, but always expose/write the latter.
        return self._normalize_rows(df, saved=True)[self.columns]

    def pre_process(self, args):
        gfx, cu_num = self.get_gfx(), self.get_cu_num()
        if gfx != "gfx950":
            self.parser.error(
                f"MXFP8 bpreshuffle tuning/replay only support gfx950; got {gfx}"
            )
        if args.splitK or args.compare or args.update_improved:
            self.parser.error(
                "Use direct GEMM tuning or --run_config; split-K/production replay are not supported"
            )
        if args.run_config:
            path = args.tune_file if args.run_config is True else args.run_config
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            args.run_config = path
            self.tunedf = self.get_tuned_gemm_list(path)
            self.untunedf = self.tunedf
            if not (self.untunedf.gfx.eq(gfx) & self.untunedf.cu_num.eq(cu_num)).all():
                raise ValueError("Saved rows do not match the current GPU's gfx/cu_num")
            return
        if not args.untune_file:
            self.parser.error("--input_file/-i is required")
        # A multi-backend tuned CSV is also a shape source. Its previous backend,
        # kid and timing are intentionally not treated as this tuner's results.
        df = self._normalize_rows(self.get_untuned_gemm_list(args.untune_file))
        df = df[df.gfx.eq(gfx) & df.cu_num.eq(cu_num)]
        if df.empty:
            raise ValueError("No input shapes match the current GPU's gfx/cu_num")
        self.untunedf = df[self.keys].drop_duplicates().reset_index(drop=True)
        for row in self.untunedf.itertuples(index=False):
            if not candidate_kids_for_shape(row.gfx, row.M, row.N, row.K):
                raise ValueError(
                    f"No MXFP8 bpreshuffle candidate supports {(row.M, row.N, row.K)}"
                )
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        if not args.all and not self.tunedf.empty:
            self.untunedf = self.untunedf[
                ~self.untunedf.apply(tuple, axis=1).isin(
                    self.tunedf[self.keys].apply(tuple, axis=1)
                )
            ].reset_index(drop=True)

    def tune(self, untunedf, tunedf, args):
        tasks, groups, requested_kids = [], [], set()
        for row in untunedf.itertuples(index=False):
            kids = candidate_kids_for_shape(row.gfx, row.M, row.N, row.K)
            if not kids:
                raise ValueError(f"No MXFP8 bpreshuffle candidates for {row}")
            requested_kids.update(kids)
            for kid in kids:
                tasks.append(
                    (
                        (
                            tuple(row),
                            kid,
                            0,
                            a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid].name,
                        ),
                        generate_data,
                        (row.M, row.N, row.K, kid),
                        run_bench,
                        (_BENCH_KEYS, kid),
                        {"num_warmup": args.warmup, "num_iters": args.iters},
                        run_torch,
                        (_REF_KEYS,),
                        {"with_bounds": True},
                        None,
                        1e-2,
                        1e-2,
                        compare_outputs,
                        None,
                        ("out",),
                    )
                )
            groups.append((len(kids), ()))
        _ensure_kids_compiled(requested_kids)
        return mp_tuner(
            tasks,
            groups,
            mp_num=args.mp,
            shape_grouped=args.shape_grouped,
            err_ratio=args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    def getKernelName(self, kernel_id):
        return a8w8_mxscale_gemm_bpreshuffle_kernels_list[kernel_id].name

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes)

    def result_to_df(self, results):
        df = super().result_to_df(results)
        df["libtype"] = "opus"
        return df[self.columns]

    def run_config(self, args):
        from aiter.test_common import run_perftest

        results = []
        rows = self._normalize_rows(self.untunedf, saved=True)
        for row in rows.itertuples(index=False):
            kid = row.kernelId
            if kid not in candidate_kids_for_shape(
                row.gfx, row.M, row.N, row.K, row.outdtype
            ):
                raise ValueError(f"Saved kid {kid} is incompatible with {row}")
        _ensure_kids_compiled(set(rows.kernelId))
        for row in rows.itertuples(index=False):
            kid = row.kernelId
            data = generate_data(row.M, row.N, row.K, kid, device="cuda")
            ref = run_torch(*(data[key] for key in _REF_KEYS), with_bounds=True)
            data["out"].fill_(float("nan"))
            out, us = run_perftest(
                run_bench,
                *(data[key] for key in _BENCH_KEYS),
                kid,
                num_warmup=args.warmup,
                num_iters=args.iters,
            )
            error = compare_outputs(
                ref, out, printLog=args.verbose, tol_err_ratio=args.errRatio
            )
            if (
                not math.isfinite(us)
                or us <= 0
                or not math.isfinite(error)
                or error > args.errRatio
            ):
                raise RuntimeError(f"Saved kid {kid} failed: {us=}, errRatio={error}")
            results.append(
                {
                    "shape": f"M={row.M},N={row.N},K={row.K},kid={kid}",
                    "e2e_us": us,
                    "errRatio": error,
                    "status": "ok",
                }
            )
        return results


def main():
    tuner = OpusMxscaleBpreshuffleTuner()
    tuner.run(tuner.parse_args())


if __name__ == "__main__":
    main()
