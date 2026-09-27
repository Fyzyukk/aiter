# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gfx950 MXFP8 B-preshuffle GEMM across OPUS and common backends.

This is deliberately kept as a gfx950/MXFP8 adapter instead of changing the
generic blockscale tuner.  Every backend receives operands derived from the
same native E8M0 dataset: OPUS consumes the compact E8M0 tensors directly,
while CK, CKTile and ASM consume exact FP32 decodes prepared outside timing.
"""

import argparse
import math
import os
import sys
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
_DEFAULT_HIP_CLANG_PATH = "/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin"
_BENCH_KEYS = ("x", "w", "out", "x_scale", "w_scale")
_REF_KEYS = ("x", "w_reference", "x_scale", "w_scale")
_CK_BENCH_KEYS = (
    "x",
    "weight_shuffle",
    "x_scale_t_fp32",
    "w_scale_fp32",
    "out",
)
_CK_ROWMAJOR_BENCH_KEYS = (
    "x",
    "weight_shuffle",
    "x_scale_fp32",
    "w_scale_fp32",
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


def _ensure_kids_compiled(candidate_kids):
    """Reuse the OPUS subset builder with the amdgpu-pin-op-dst compiler."""
    candidate_kids = frozenset(int(kid) for kid in candidate_kids)
    if not candidate_kids:
        return False

    compiler_path = os.environ.get("OPUS_HIP_CLANG_PATH", _DEFAULT_HIP_CLANG_PATH)
    if not Path(compiler_path).is_dir():
        raise FileNotFoundError(
            "The 4-wave MXFP8 kernel requires the yuyzhang512/llvm-project "
            "amdgpu-pin-op-dst toolchain; "
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
        if not a8w8_mxscale_bpreshuffle_supports_shape(instance, m, n, k):
            continue
        candidates.append(kid)
    return candidates


def generate_data(m, n, k, seed=0, *, device):
    """Build one native-E8M0 dataset shared by every selected backend.

    ``x_scale``/``w_scale`` retain the compact tensors required by OPUS.  The
    ``*_fp32`` entries are exact decodes of those tensors for the existing
    CK/CKTile/ASM blockscale interfaces.  All conversion and B preshuffling is
    therefore outside the timed call.
    """
    if min(m, n, k) <= 0 or n % 16 or k % _SCALE_GROUP_K:
        raise ValueError(
            f"MXFP8 B-preshuffle data requires positive shapes, N divisible by "
            f"16 and K divisible by {_SCALE_GROUP_K}; got {(m, n, k)}"
        )
    # Keep the original standalone tuner's deterministic dataset.  ``seed`` is
    # accepted because the generic tuner includes it in every task signature.
    generator = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((m, k), device=device, generator=generator).to(torch.float8_e4m3fn)
    weight = torch.randn((n, k), device=device, generator=generator).to(x.dtype)
    kg = k // _SCALE_GROUP_K
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
        (math.ceil(n / _SCALE_GROUP_N), kg),
        device=device,
        dtype=torch.uint8,
        generator=generator,
    ).view(torch.float8_e8m0fnu)
    x_scale_fp32 = x_scale.float().contiguous()
    w_scale_fp32 = w_scale.float().contiguous()
    x_scale_t_fp32 = x_scale_fp32.transpose(0, 1).contiguous().view(*x_scale_fp32.shape)
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    return {
        "x": x,
        # Existing OPUS names are retained for direct tuner/replay callers.
        "w": weight_shuffle,
        "w_reference": weight,
        "out": torch.empty((m, n), device=device, dtype=torch.bfloat16),
        "x_scale": x_scale,
        "w_scale": w_scale,
        # Generic blockscale backend names, all derived from the same values.
        "weight": weight,
        "weight_shuffle": weight_shuffle,
        "x_scale_fp32": x_scale_fp32,
        "x_scale_t_fp32": x_scale_t_fp32,
        "w_scale_fp32": w_scale_fp32,
        "zero_bias": torch.zeros((1, n), dtype=torch.float32, device=device),
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
        "errRatio": 0.0,
        # Exact replay calls backend tuning entries directly. Publishing this
        # dataset as a production FP32-scale config would change its contract.
        "config_env_name": None,
    }

    def __init__(self):
        self.opus_kids = None
        super().__init__(
            "opus_mxscale_bpreshuffle",
            # Input/output/scale dtypes are fixed by this specialized tuner.
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
                "OPUS using native E8M0 A 1x128 / B 128x128 scales and BF16 "
                "output. Measures backend GPU time, including internal transforms."
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
                    "(or -o when omitted), with native E8M0 inputs; measures backend GPU time"
                )
            elif action.dest == "splitK":
                action.help = (
                    "include supported ASM split-K candidates; CK, CKTile and "
                    "OPUS B-preshuffle candidates use splitK=0"
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
        if (
            not df["scale_dtype"]
            .isin(["e8m0", "float8_e8m0fnu", str(torch.float8_e8m0fnu)])
            .all()
        ):
            raise ValueError("MXFP8 bpreshuffle tune requires E8M0 scales")
        if saved and (df["libtype"].isna() | df["libtype"].astype(str).eq("")).any():
            raise ValueError("Saved MXFP8 rows require a non-empty libtype")
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
        kids = candidate_kids_for_shape(gfx, m, n, k)
        return (
            kids
            if self.opus_kids is None
            else [kid for kid in kids if kid in self.opus_kids]
        )

    @staticmethod
    def _make_task(info, m, n, k, seed, func, arg_keys, extra_args, run_kwargs):
        return (
            info,
            generate_data,
            (m, n, k, seed),
            func,
            (arg_keys, *extra_args),
            dict(run_kwargs),
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

    @classmethod
    def _adapt_generic_tasks(cls, tasks, info_keys, seed):
        """Retarget generic tasks to the shared native-E8M0 dataset."""
        _gfx, _cu_num, m, n, k = info_keys
        key_map = {
            "x_scale": "x_scale_fp32",
            "x_scale_t": "x_scale_t_fp32",
            "w_scale": "w_scale_fp32",
        }
        adapted = []
        for task in tasks:
            info, _gen_data, _gen_args, func, args, kwargs, *_rest = task
            arg_keys = tuple(key_map.get(key, key) for key in args[0])
            if info[4] == "cktile":
                arg_keys = cktile_bench_keys(info[1])
            adapted.append(
                cls._make_task(
                    info,
                    m,
                    n,
                    k,
                    seed,
                    func,
                    arg_keys,
                    args[1:],
                    kwargs,
                )
            )
        return adapted

    def get_gemm_a8w8_blockscale_tune_task(
        self, info_keys, useSplitK, seed, preshuffleB, run_kwargs
    ):
        tasks = super().get_gemm_a8w8_blockscale_tune_task(
            info_keys, useSplitK, seed, preshuffleB, run_kwargs
        )
        return self._adapt_generic_tasks(tasks, info_keys, seed)

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
        return self._adapt_generic_tasks(tasks, info_keys, seed)

    def get_gemm_a8w8_blockscale_asm_tune_task(
        self, info_keys, useSplitK, seed, preshuffleB, run_kwargs
    ):
        tasks = super().get_gemm_a8w8_blockscale_asm_tune_task(
            info_keys, useSplitK, seed, preshuffleB, run_kwargs
        )
        return self._adapt_generic_tasks(tasks, info_keys, seed)

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
        return super().tune(untunedf, tunedf, args)

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
        from aiter.test_common import run_perftest

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
                    row.gfx, row.M, row.N, row.K, row.outdtype
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
            data = generate_data(row.M, row.N, row.K, 0, device="cuda")
            ref = run_torch(*(data[key] for key in _REF_KEYS), with_bounds=True)
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
            error = compare_outputs(
                ref, out, printLog=args.verbose, tol_err_ratio=args.errRatio
            )
            if (
                not math.isfinite(us)
                or us <= 0
                or not math.isfinite(error)
                or error > args.errRatio
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
