# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gfx950 MXFP8 B-preshuffle GEMM across OPUS and common backends.

This gfx950 adapter keeps each backend's scale input format: OPUS receives
random native E8M0 scales, while CK, CKTile and ASM reuse the original
blockscale tuner's random FP32 scales, reference and accuracy checks.
"""

import argparse
import math
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import ClassVar

import pandas as pd
import torch

import aiter
from aiter import logger
from aiter.utility import dtypes

# AOT import omits these top-level aliases, but the shared tuner uses them.
aiter.dtypes = dtypes
from aiter.ops import gemm_op_a8w8

for _entry in (
    "gemm_a8w8_blockscale_bpreshuffle_tune",
    "gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
    "gemm_a8w8_blockscale_bpreshuffle_asm",
):
    setattr(aiter, _entry, getattr(gemm_op_a8w8, _entry))

from aiter.ops.opus import opus_gemm
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from csrc.opus_gemm.opus_gemm_common import (
    A8W8_BPRESHUFFLE_TUNING_KIDS,
    a8w8_mxscale_bpreshuffle_supports_shape,
    a8w8_mxscale_gemm_bpreshuffle_kernels_list,
    canonical_output_dtype,
)

# The generic tuner is directly executable and imports its sibling instance
# modules by name.  Make that directory importable before loading it, then
# reuse its backend runners, candidate registries and tuner behavior here.
_CK_TUNER_DIR = Path(__file__).resolve().parents[1] / "ck_gemm_a8w8_blockscale"
if str(_CK_TUNER_DIR) not in sys.path:
    sys.path.insert(0, str(_CK_TUNER_DIR))
from csrc.ck_gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale_tune as generic_tune,
)

_TAG = "a8w8_mxscale_gemm_bpreshuffle"
# yuyzhang512/llvm-project, amdgpu-pin-op-dst at 49c418896816.
# The imported pipeline requires statement-level destination pinning.
_DEFAULT_HIP_CLANG_PATH = "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin"
_BENCH_KEYS = ("x", "w", "out", "x_scale", "w_scale")
_REF_KEYS = ("x", "w_reference", "x_scale", "w_scale")
_CK_REF_KEYS = ("x", "weight", "x_scale", "w_scale")
_CK_BENCH_KEYS = (
    "x",
    "weight_shuffle",
    "x_scale_t",
    "w_scale",
    "out",
)
_CK_ROWMAJOR_BENCH_KEYS = (
    "x",
    "weight_shuffle",
    "x_scale",
    "w_scale",
    "out",
)
_ASM_BENCH_KEYS = (*_CK_BENCH_KEYS, "zero_bias")
_SCALE_GROUP_N = 128
_SCALE_GROUP_K = 128
_SUPPORTED_LIBTYPES = frozenset(("ck", "cktile", "asm", "opus"))


def parse_opus_kids(value):
    """Parse an optional registry subset without changing exact saved replay."""
    try:
        kids = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--opus-kids requires comma-separated integer IDs"
        ) from exc
    if not kids or len(kids) != len(set(kids)) or min(kids) < 0:
        raise argparse.ArgumentTypeError("--opus-kids requires unique nonnegative IDs")
    return frozenset(kids)


def cktile_bench_keys(kid):
    """Match the CKTile wrapper's per-instance activation-scale contract.

    Eight-wave AQRowMajor instances consume logical [M,K/128] row-major
    storage directly. Other B-preshuffle instances accept physical [K/128,M]
    storage; the four-wave wrapper converts it back to row-major internally.
    """
    kernel = generic_tune.candidate_kernels_cktile_dict[kid]
    if kernel.is_eight_warp and kernel.AQRowMajor:
        return _CK_ROWMAJOR_BENCH_KEYS
    return _CK_BENCH_KEYS


@contextmanager
def _opus_compiler_environment():
    """Build OPUS with its pin LLVM, optionally mixing a baseline LLVM."""
    from aiter.jit import core
    from cpp_extension import ROCM_HOME

    compiler_path = os.environ.get("OPUS_HIP_CLANG_PATH", _DEFAULT_HIP_CLANG_PATH)
    compiler = Path(compiler_path) / "clang++"
    if not compiler.is_file():
        raise FileNotFoundError(
            "The 4-wave MXFP8 kernel requires the yuyzhang512/llvm-project "
            "amdgpu-pin-op-dst toolchain; "
            f"directory not found: {compiler_path}. Set OPUS_HIP_CLANG_PATH "
            "to its bin directory."
        )
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    match = re.search(r"clang version (\d+)", version)
    if match is None:
        raise RuntimeError(f"Cannot identify the selected LLVM: {compiler}")
    resource_dir = os.environ.get("OPUS_HIP_RESOURCE_DIR") or os.environ.get(
        "AITER_HIP_RESOURCE_DIR"
    )
    if not resource_dir and int(match[1]) >= 24 and core.get_hip_version().startswith("7.0."):
        # LLVM 24 removed half OCML declarations that ROCm 7.0 HIP headers use.
        candidates = [Path(ROCM_HOME) / suffix for suffix in (
            "lib/llvm/lib/clang/20", "llvm/lib/clang/20",
        )] if ROCM_HOME else []
        resource_dir = next((str(path) for path in candidates if path.is_dir()), None)
        if resource_dir is None:
            raise RuntimeError(
                "LLVM 24 with ROCm 7.0 requires OPUS_HIP_RESOURCE_DIR pointing "
                "to the ROCm clang resource headers"
            )
    if resource_dir:
        if not Path(resource_dir, "include").is_dir():
            raise FileNotFoundError(f"Invalid OPUS HIP resource directory: {resource_dir}")

    baseline_path = os.environ.get("OPUS_BASELINE_HIP_CLANG_PATH")
    build_compiler = compiler_path
    build_resources = resource_dir
    if baseline_path:
        if not Path(baseline_path, "clang++").is_file():
            raise FileNotFoundError(f"Invalid OPUS_BASELINE_HIP_CLANG_PATH: {baseline_path}")
        build_compiler = baseline_path
        build_resources = os.environ.get("OPUS_BASELINE_HIP_RESOURCE_DIR")
        if build_resources and not Path(build_resources, "include").is_dir():
            raise FileNotFoundError(f"Invalid OPUS baseline resources: {build_resources}")

    previous = {
        key: os.environ.get(key) for key in (
            "HIP_CLANG_PATH", "AITER_HIP_RESOURCE_DIR", "OPUS_HIP_CLANG_PATH",
            "OPUS_HIP_RESOURCE_DIR",
        )
    }
    try:
        os.environ["OPUS_HIP_CLANG_PATH"] = compiler_path
        if resource_dir:
            os.environ["OPUS_HIP_RESOURCE_DIR"] = resource_dir
        os.environ["HIP_CLANG_PATH"] = build_compiler
        if build_resources:
            os.environ["AITER_HIP_RESOURCE_DIR"] = build_resources
        else:
            os.environ.pop("AITER_HIP_RESOURCE_DIR", None)
        core.hip_flag_checker.cache_clear()
        core.check_LLVM_MAIN_REVISION.cache_clear()
        logger.info(
            "MXFP8 OPUS pin compiler: "
            f"{version.splitlines()[0]}; baseline={baseline_path or 'same compiler'}; "
            f"pin resources={resource_dir or 'compiler default'}"
        )
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        # The probes are cached without a compiler key. Do not let results
        # from the pin branch affect subsequent CK/CKTile/ASM wrapper builds.
        core.hip_flag_checker.cache_clear()
        core.check_LLVM_MAIN_REVISION.cache_clear()


def _ensure_kids_compiled(candidate_kids):
    """Reuse the OPUS subset builder with the amdgpu-pin-op-dst compiler."""
    candidate_kids = frozenset(int(kid) for kid in candidate_kids)
    if not candidate_kids:
        return False

    # opus_gemm_tune.py is also a directly executable script and therefore
    # uses sibling absolute imports. Load it lazily only when tuning/replaying,
    # after making that sibling directory importable.
    opus_dir = str(Path(__file__).resolve().parent)
    added_path = opus_dir not in sys.path
    if added_path:
        sys.path.insert(0, opus_dir)
    try:
        from opus_gemm_tune import _ensure_kids_compiled as ensure_existing_opus_kids

        with _opus_compiler_environment():
            return ensure_existing_opus_kids(candidate_kids)
    finally:
        if added_path:
            sys.path.remove(opus_dir)


def candidate_kids_for_shape(gfx, m, n, k, outdtype="bf16", *, include_legacy=False):
    """Use registry tiling and byte limits before allocating tuning inputs."""
    if min(m, n, k) <= 0:
        return []
    canonical_out = canonical_output_dtype(outdtype)
    out_bytes = {"bf16_t": 2, "fp32_t": 4}.get(canonical_out)
    if out_bytes is None:
        return []
    candidates = []
    for kid, instance in sorted(a8w8_mxscale_gemm_bpreshuffle_kernels_list.items()):
        if not include_legacy and kid not in A8W8_BPRESHUFFLE_TUNING_KIDS:
            continue
        if (
            gfx != (instance.arch_prefix or "gfx950")
            or instance.kernel_tag != _TAG
            or canonical_out not in instance.output_dtypes
        ):
            continue
        if not a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, k):
            continue
        candidates.append(kid)
    return candidates


def generate_data(m, n, k, seed=0, *, device):
    """Build OPUS inputs with random FP8 operands and native E8M0 scales."""
    if min(m, n, k) <= 0 or n % 16 or k % _SCALE_GROUP_K:
        raise ValueError(
            f"MXFP8 B-preshuffle data requires positive shapes, N divisible by "
            f"16 and K divisible by {_SCALE_GROUP_K}; got {(m, n, k)}"
        )
    # Match the original tuner's operand distribution; scales use OPUS's
    # native format instead of the generic tuner's independently random FP32.
    generator = torch.Generator(device=device).manual_seed(seed)
    x = (
        torch.rand((m, k), dtype=torch.float16, device=device, generator=generator)
        / 10
    ).to(torch.float8_e4m3fn)
    weight = (
        torch.rand((n, k), dtype=torch.float16, device=device, generator=generator)
        / 10
    ).to(x.dtype)
    kg = k // _SCALE_GROUP_K
    # Generate finite powers of two directly in the native exponent storage.
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
        (math.ceil(n / _SCALE_GROUP_N), kg),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    ).view(torch.float8_e8m0fnu)
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    return {
        "x": x,
        "w": weight_shuffle,
        "w_reference": weight,
        "out": torch.empty((m, n), device=device, dtype=torch.bfloat16),
        "x_scale": x_scale,
        "w_scale": w_scale,
    }


def run_torch(x, w, x_scale, w_scale, *, with_bounds=False):
    """Independent FP32 reference and optional absolute-product magnitudes."""
    a_scale = x_scale.float().repeat_interleave(_SCALE_GROUP_K, 1)[:, : x.shape[1]]
    b_scale = (
        w_scale.float()
        .repeat_interleave(_SCALE_GROUP_N, 0)
        .repeat_interleave(_SCALE_GROUP_K, 1)[: w.shape[0], : w.shape[1]]
    )
    a = x.float() * a_scale
    b = w.float() * b_scale
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


class OpusMxscaleBpreshuffleTuner(generic_tune.GemmA8W8BlockScaleTuner):
    ARG_DEFAULTS: ClassVar[dict] = {
        **generic_tune.GemmA8W8BlockScaleTuner.ARG_DEFAULTS,
        "tune_file": "/tmp/opus_mxscale_bpreshuffle_tuned.csv",
        "untune_file": "",
        # Inherit the external backends' original error-ratio threshold.
        # OPUS candidates retain their stricter zero-outlier contract.
        # Exact replay calls backend tuning entries directly. Publishing this
        # dataset as a production FP32-scale config would change its contract.
        "config_env_name": None,
    }

    def __init__(self):
        self.opus_kids = None
        super().__init__(
            "opus_mxscale_bpreshuffle",
            # Input/output dtypes are fixed; scale dtype follows the backend.
            # Keep them as validated internal defaults, not serialized shape
            # keys, so -o matches the production tuned-config CSV schema.
            keys=["gfx", "cu_num", "M", "N", "K"],
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
            description=(
                "Tune gfx950 FP8 B-preshuffle GEMM across CK, CKTile, ASM and "
                "OPUS with BF16 output. CK/CKTile/ASM use random FP32 scales; "
                "OPUS uses random native E8M0 scales. Measures backend GPU "
                "time, including internal transforms."
            ),
        )

    def _setup_specific_arguments(self):
        super()._setup_specific_arguments()
        self.parser.add_argument(
            "--opus-kids",
            type=parse_opus_kids,
            default=None,
            help="comma-separated registered OPUS IDs to tune; default: all legal OPUS; ignored for --run_config",
        )
        # This adapter is B-preshuffle-only.  Keep the generic flag accepted so
        # existing tune commands remain valid, but make it true by default.
        self.parser.set_defaults(preshuffle=True)
        for action in self.parser._actions:
            if action.dest == "libtype":
                action.help = (
                    "backend candidates to tune: ck, cktile, asm, opus, both "
                    "(CK + CKTile), or all"
                )
            elif action.dest == "preshuffle":
                action.help = "accepted for compatibility; always enabled by this tuner"
            elif action.dest == "run_config":
                action.help = (
                    "replay saved exact backend/kernel choices from TUNED_CSV "
                    "(or -o when omitted), with each backend's scale format; measures backend GPU time"
                )
            elif action.dest == "splitK":
                action.help = (
                    "include supported ASM split-K candidates; CK, CKTile and "
                    "OPUS B-preshuffle candidates use splitK=0"
                )
            elif action.dest == "errRatio":
                action.help = (
                    "maximum outlier fraction for the original CK/CKTile/ASM "
                    "accuracy checks (default: 0.05); OPUS always requires "
                    "zero outliers under its accumulation-bounds check"
                )
        self.parser.add_argument(
            "--input_file", dest="untune_file", default=argparse.SUPPRESS
        )
        self.parser.add_argument(
            "--tuned_file", dest="tune_file", default=argparse.SUPPRESS
        )

    def run(self, args, fast_mode=False):
        # Avoid the generic tuner's class-level default mutation; this adapter
        # already has a fixed B-preshuffle output/config family. Exact replay
        # must also retain different backends for the same shape in -o2 files,
        # without triggering the production replay's config/JIT-cache updates.
        args.preshuffle = True
        if args.run_config:
            self.pre_process(args)
            results = self.run_config(args)
            self._print_benchmark_results(
                "MXFP8 exact replay (backend GPU time)", results
            )
            return self.tunedf
        return GemmCommonTuner.run(self, args, fast_mode)

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
        if (df["N"] % 16 != 0).any() or (df["K"] % _SCALE_GROUP_K != 0).any():
            raise ValueError(
                "MXFP8 B-preshuffle requires N divisible by 16 and K divisible by 128"
            )
        if (
            not df["dtype"]
            .isin(["fp8", "float8_e4m3fn", str(torch.float8_e4m3fn)])
            .all()
        ):
            raise ValueError("MXFP8 bpreshuffle tune requires FP8 E4M3FN inputs")
        if not df["outdtype"].map(canonical_output_dtype).eq("bf16_t").all():
            raise ValueError("MXFP8 bpreshuffle tune requires BF16 output")
        if saved and (df["libtype"].isna() | df["libtype"].astype(str).eq("")).any():
            raise ValueError("Saved MXFP8 rows require a non-empty libtype")
        df["dtype"], df["outdtype"] = (
            str(torch.float8_e4m3fn),
            str(torch.bfloat16),
        )
        # Scale dtype is selected by the backend, never by a legacy CSV column.
        if saved:
            df["scale_dtype"] = df["libtype"].eq("opus").map(
                {True: "e8m0", False: "fp32"}
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
        args.preshuffle = True
        self.opus_kids = None
        if args.compare or args.update_improved:
            self.parser.error(
                "Use direct GEMM tuning or --run_config; production compare/update "
                "does not yet preserve native E8M0 scales"
            )
        if args.run_config:
            path = args.tune_file if args.run_config is True else args.run_config
            if not Path(path).is_file():
                raise FileNotFoundError(path)
            args.run_config = path
            self.tunedf = self.get_tuned_gemm_list(path)
            self.untunedf = self.tunedf[
                self.tunedf.gfx.eq(gfx) & self.tunedf.cu_num.eq(cu_num)
            ].reset_index(drop=True)
            if self.untunedf.empty:
                raise ValueError("No saved rows match the current GPU's gfx/cu_num")
            return
        if not args.untune_file:
            self.parser.error("--input_file/-i is required")
        self.opus_kids = getattr(args, "opus_kids", None)
        if self.opus_kids is not None:
            unknown = self.opus_kids.difference(
                a8w8_mxscale_gemm_bpreshuffle_kernels_list
            )
            if unknown:
                self.parser.error(
                    f"Unknown MXFP8 B-preshuffle OPUS IDs: {sorted(unknown)}"
                )
        # A multi-backend tuned CSV is also a shape source. Its previous backend,
        # kid and timing are intentionally not treated as this tuner's results.
        df = self._normalize_rows(self.get_untuned_gemm_list(args.untune_file))
        df = df[df.gfx.eq(gfx) & df.cu_num.eq(cu_num)]
        if df.empty:
            raise ValueError("No input shapes match the current GPU's gfx/cu_num")
        self.untunedf = df[self.keys].drop_duplicates().reset_index(drop=True)
        if args.libtype == "opus":
            for row in self.untunedf.itertuples(index=False):
                if not self._candidate_kids(row.gfx, row.M, row.N, row.K):
                    raise ValueError(
                        f"No MXFP8 bpreshuffle candidate supports "
                        f"{(row.M, row.N, row.K)}"
                    )
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        if not args.all and not self.tunedf.empty:
            self.untunedf = self.untunedf[
                ~self.untunedf.apply(tuple, axis=1).isin(
                    self.tunedf[self.keys].apply(tuple, axis=1)
                )
            ].reset_index(drop=True)

    def _candidate_kids(self, gfx, m, n, k):
        kids = candidate_kids_for_shape(
            gfx, m, n, k, include_legacy=self.opus_kids is not None,
        )
        return (
            kids
            if self.opus_kids is None
            else [kid for kid in kids if kid in self.opus_kids]
        )

    @staticmethod
    def _make_task(
        info,
        m,
        n,
        k,
        seed,
        func,
        arg_keys,
        extra_args,
        run_kwargs,
        *,
        gen_data=generate_data,
        ref_keys=_REF_KEYS,
    ):
        return (
            info,
            gen_data,
            (m, n, k, seed),
            func,
            (arg_keys, *extra_args),
            dict(run_kwargs),
            run_torch,
            (ref_keys,),
            {"with_bounds": True},
            None,
            1e-2,
            1e-2,
            compare_outputs,
            None,
            ("out",),
        )

    @staticmethod
    def _adapt_generic_tasks(tasks):
        """Preserve the external tuner's data, reference and comparison contract."""
        adapted = []
        for task in tasks:
            info, args = task[0], task[4]
            if info[4] == "cktile":
                # The wrapper's activation-scale layout depends on the instance.
                args = (cktile_bench_keys(info[1]), *args[1:])
                task = (*task[:4], args, *task[5:])
            adapted.append(task)
        return adapted

    def get_gemm_a8w8_blockscale_tune_task(
        self, info_keys, useSplitK, seed, preshuffleB, run_kwargs
    ):
        tasks = super().get_gemm_a8w8_blockscale_tune_task(
            info_keys, useSplitK, seed, preshuffleB, run_kwargs
        )
        return self._adapt_generic_tasks(tasks)

    def get_gemm_a8w8_blockscale_cktile_tune_task(
        self,
        info_keys,
        useSplitK,
        seed,
        preshuffleB,
        block_per_cu,
        run_kwargs,
    ):
        tasks = super().get_gemm_a8w8_blockscale_cktile_tune_task(
            info_keys,
            useSplitK,
            seed,
            preshuffleB,
            block_per_cu,
            run_kwargs,
        )
        return self._adapt_generic_tasks(tasks)

    def get_gemm_a8w8_blockscale_asm_tune_task(
        self, info_keys, useSplitK, seed, preshuffleB, run_kwargs
    ):
        tasks = super().get_gemm_a8w8_blockscale_asm_tune_task(
            info_keys, useSplitK, seed, preshuffleB, run_kwargs
        )
        return self._adapt_generic_tasks(tasks)

    def get_gemm_a8w8_blockscale_opus_tune_task(
        self, info_keys, seed, preshuffleB, run_kwargs
    ):
        gfx, _cu_num, m, n, k = info_keys
        if not preshuffleB:
            return []
        tasks = []
        for kid in self._candidate_kids(gfx, m, n, k):
            kernel = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
            info = (info_keys, kid, 0, kernel.name, "opus", True)
            tasks.append(
                self._make_task(
                    info,
                    m,
                    n,
                    k,
                    seed,
                    run_bench,
                    _BENCH_KEYS,
                    (kid,),
                    run_kwargs,
                )
            )
        return tasks

    def tune(self, untunedf, tunedf, args):
        self.opus_kids = getattr(args, "opus_kids", None)
        requested_kids = set()
        if args.libtype in ("opus", "all"):
            for row in untunedf.itertuples(index=False):
                kids = self._candidate_kids(row.gfx, row.M, row.N, row.K)
                if args.libtype == "opus" and not kids:
                    raise ValueError(
                        f"No OPUS MXFP8 bpreshuffle candidates for "
                        f"{(row.M, row.N, row.K)}"
                    )
                requested_kids.update(kids)
        if requested_kids:
            _ensure_kids_compiled(requested_kids)

        tasks, tasks_data = [], []
        run_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}
        gfx, cu_num = self.get_gfx(), self.get_cu_num()
        for row in untunedf.itertuples(index=False):
            info = (gfx, cu_num, row.M, row.N, row.K)
            start = len(tasks)
            if args.libtype in ("ck", "both", "all"):
                tasks.extend(self.get_gemm_a8w8_blockscale_tune_task(
                    info, args.splitK, 0, True, run_kwargs
                ))
            if args.libtype in ("cktile", "both", "all"):
                tasks.extend(self.get_gemm_a8w8_blockscale_cktile_tune_task(
                    info, args.splitK, 0, True, args.blockPerCu, run_kwargs
                ))
            if args.libtype in ("asm", "all"):
                tasks.extend(self.get_gemm_a8w8_blockscale_asm_tune_task(
                    info, args.splitK, 0, True, run_kwargs
                ))
            if args.libtype in ("opus", "all"):
                tasks.extend(self.get_gemm_a8w8_blockscale_opus_tune_task(
                    info, 0, True, run_kwargs
                ))
            if len(tasks) != start:
                tasks_data.append((len(tasks) - start, ()))
        if not tasks:
            return []

        # In the unchanged mp_tuner, fast_mode=True refreshes the reference
        # when the generator changes. Every task supplies its own reference;
        # external tasks retain checkAllclose and OPUS uses compare_outputs.
        return generic_tune.mp_tuner(
            tasks, tasks_data, args.mp, True,
            args.shape_grouped or args.mp == 1, args.errRatio,
            timeout=args.timeout, verbose=args.verbose,
        )

    @staticmethod
    def _error_limit(libtype, args):
        return 0.0 if libtype == "opus" else args.errRatio

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        rets = list(rets)
        # Save the original timings and measured error ratios without alteration.
        raw = super().post_process(rets, args, topk=-1, fast_mode=True)
        if fast_mode or topk == -1:
            return raw
        # The shared selector has one threshold. Mark rejected candidates as
        # unavailable for selection while retaining their raw profile records.
        selection = [
            (
                info,
                us if math.isfinite(error)
                and 0 <= error <= self._error_limit(info[4], args)
                else self.INVALID_TIME,
                error,
            )
            for info, us, error in rets
        ]
        selection_args = argparse.Namespace(**vars(args))
        selection_args.profile_file = ""
        return super().post_process(selection, selection_args, topk, False)

    def getKernelName(self, kernel_id, libType="opus", preshuffleB=True):
        if libType == "opus":
            kernel = a8w8_mxscale_gemm_bpreshuffle_kernels_list.get(kernel_id)
            return None if kernel is None else kernel.name
        return super().getKernelName(kernel_id, libType, preshuffleB)

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes)

    def result_to_df(self, results):
        # Accept the original OPUS-only four-field info tuple as well as the
        # generic tuner's six-field multi-backend tuple.
        normalized = []
        for info, time, err_ratio in results:
            if len(info) == 4:
                info = (*info, "opus", True)
            normalized.append((info, time, err_ratio))
        return super().result_to_df(normalized)[self.columns]

    def run_config(self, args):
        from aiter.test_common import checkAllclose, run_perftest

        results = []
        rows = self._normalize_rows(self.untunedf, saved=True)
        unsupported = set(rows.libtype).difference(_SUPPORTED_LIBTYPES)
        if unsupported:
            raise ValueError(
                f"Saved MXFP8 replay does not support libtype(s): {sorted(unsupported)}"
            )
        opus_rows = rows[rows.libtype.eq("opus")]
        for row in rows.itertuples(index=False):
            if row.gfx != self.get_gfx() or row.cu_num != self.get_cu_num():
                raise ValueError(
                    "Saved row does not match the current GPU's gfx/cu_num"
                )
            if row.libtype == "opus" and (
                row.splitK != 0
                or row.kernelId
                not in candidate_kids_for_shape(
                    row.gfx, row.M, row.N, row.K, row.outdtype, include_legacy=True,
                )
            ):
                raise ValueError(
                    f"Saved OPUS kid {row.kernelId} is incompatible with {row}"
                )
            if (
                row.libtype == "ck"
                and row.kernelId not in generic_tune.candidate_kernels_bpreshuffle_dict
            ):
                raise ValueError(f"Saved CK kid {row.kernelId} is incompatible")
            if (
                row.libtype == "cktile"
                and row.kernelId not in generic_tune.candidate_kernels_cktile_dict
            ):
                raise ValueError(f"Saved CKTile kid {row.kernelId} is incompatible")
            if row.libtype in ("ck", "cktile") and row.splitK != 0:
                raise ValueError("CK/CKTile B-preshuffle replay requires splitK=0")
            saved_name = getattr(row, "kernelName", None)
            if row.libtype == "asm":
                if pd.isna(saved_name) or not str(saved_name).strip():
                    raise ValueError("Saved ASM row requires kernelName")
                info_keys = tuple(getattr(row, key) for key in self.keys)
                candidates = self.get_gemm_a8w8_blockscale_asm_tune_task(
                    info_keys, True, 0, True, {}
                )
                if not any(
                    task[0][3] == saved_name and task[0][2] == row.splitK
                    for task in candidates
                ):
                    raise ValueError(
                        f"Saved ASM kernel {saved_name} / splitK={row.splitK} "
                        "is incompatible with the current B-preshuffle candidates"
                    )
            elif pd.notna(saved_name) and str(saved_name).strip():
                expected_name = self.getKernelName(row.kernelId, row.libtype, True)
                if saved_name != expected_name:
                    raise ValueError(
                        f"Saved {row.libtype} kernelName does not match kid {row.kernelId}"
                    )
        if not opus_rows.empty:
            _ensure_kids_compiled(set(opus_rows.kernelId))
        for row in rows.itertuples(index=False):
            kid = row.kernelId
            if row.libtype == "opus":
                gen_data, ref_keys = generate_data, _REF_KEYS
            else:
                gen_data, ref_keys = generic_tune.generate_data, _CK_REF_KEYS
            data = gen_data(row.M, row.N, row.K, 0, device="cuda")
            ref_inputs = tuple(data[key] for key in ref_keys)
            ref = (
                run_torch(*ref_inputs, with_bounds=True)
                if row.libtype == "opus"
                else generic_tune.run_torch(*ref_inputs)
            )
            data["out"].fill_(float("nan"))
            if row.libtype == "opus":
                bench, bench_args = run_bench, (
                    *(data[key] for key in _BENCH_KEYS),
                    kid,
                )
            elif row.libtype == "ck":
                bench, bench_args = generic_tune.run_gemm_a8w8_blockscale, (
                    *(data[key] for key in _CK_BENCH_KEYS),
                    kid,
                    row.splitK,
                    True,
                )
            elif row.libtype == "cktile":
                bench, bench_args = generic_tune.run_gemm_a8w8_blockscale_cktile, (
                    *(data[key] for key in cktile_bench_keys(kid)),
                    kid,
                    row.splitK,
                    True,
                )
            else:
                bench, bench_args = generic_tune.run_gemm_a8w8_blockscale_asm, (
                    *(data[key] for key in _ASM_BENCH_KEYS),
                    row.kernelName,
                    row.splitK,
                    True,
                )
            out, us = run_perftest(
                bench,
                *bench_args,
                num_warmup=args.warmup,
                num_iters=args.iters,
            )
            if row.libtype == "opus":
                error = compare_outputs(ref, out, printLog=args.verbose)
            else:
                error = checkAllclose(
                    ref, out, rtol=1e-2, atol=1e-2,
                    printLog=args.verbose, tol_err_ratio=args.errRatio,
                    catastrophic_check=True,
                )
            if (
                not math.isfinite(us)
                or us <= 0
                or not math.isfinite(error)
                or error < 0
                or error > self._error_limit(row.libtype, args)
            ):
                raise RuntimeError(
                    f"Saved {row.libtype} kid {kid} failed: {us=}, errRatio={error}"
                )
            results.append(
                {
                    "shape": (
                        f"M={row.M},N={row.N},K={row.K},"
                        f"libtype={row.libtype},kid={kid}"
                    ),
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
