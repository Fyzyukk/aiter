# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune and compare gfx950 MXFP8 B-preshuffle OPUS and FlyDSL BMM.

OPUS and FlyDSL BMM share FP8 operands and native E8M0 scales: A 1x128 and
B 128x128. Both consume the same preshuffled B storage; BMM adds batch=1
views. The default --libtype both searches both backends on the same GPU
per shape and immediately writes their best valid timings to one CSV.
Single-backend tuning remains available. No scale conversion is performed.
"""

import argparse
import csv
import functools
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import ClassVar


def _explain_candidates(argv=None):
    """Report candidates before importing the runtime tuner dependencies."""
    if __package__:
        from .opus_gemm_common import explain_selection
    else:
        from opus_gemm_common import explain_selection

    parser = argparse.ArgumentParser(description="Explain gfx950 B-preshuffle candidates without loading the GPU runtime.")
    parser.add_argument("--explain-candidates", action="store_true", help="report candidates without running GPU tuning")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("M", "N", "K"), required=True)
    parser.add_argument("--pipelines", help="comma-separated pipelines; panel_2wave is opt-in")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--per-pipeline", type=int, default=2)
    parser.add_argument("--cu-num", type=int, default=256, help="target CU count for grid priority (no device query)")
    parser.add_argument("--explain", action="store_true", help="include rejected and lower-priority candidates")
    args = parser.parse_args(argv)
    try:
        report = explain_selection(*args.shape, pipelines=args.pipelines,
                                   max_candidates=args.max_candidates, per_pipeline=args.per_pipeline,
                                   cu_num=args.cu_num)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.explain:
        report["candidates"] = [row for row in report["candidates"] if row["selected"]]
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__" and "--explain-candidates" in sys.argv[1:]:
    _explain_candidates()
    raise SystemExit(0)


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
import aiter.utility.mp_tuner as mp_workers
from csrc.opus_gemm.opus_gemm_common import (
    CONFIG_COLUMNS,
    a8w8_mxscale_gemm_bpreshuffle_kernels_list,
    candidate_kids_for_shape,
    canonical_output_dtype,
    configuration_metadata,
    bpreshuffle_bmm_candidates,
    bpreshuffle_workspace_bytes,
)

# The generic tuner is directly executable and imports its sibling instance
# modules by name.  Make that directory importable before loading it, then
# reuse its task scheduler, result serialization and tuner behavior here.
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
_SCALE_GROUP_N = 128
_SCALE_GROUP_K = 128
_OPUS_LIBTYPES = frozenset(("opus",))
_SUPPORTED_LIBTYPES = _OPUS_LIBTYPES | {"flydsl"}
_COMPARISON_COLUMNS = (
    "opus_backend", "opus_kernelId", "opus_splitK", "opus_us",
    "opus_kernelName", "opus_errRatio",
    *("opus_" + column for column in CONFIG_COLUMNS),
    "flydsl_kernelId", "flydsl_splitK", "flydsl_us",
    "flydsl_kernelName", "flydsl_errRatio",
    "winner", "BMM_over_OPUS", "OPUS_speedup_pct", "comparison_status",
)


def tune_shape_groups(tasks, in_datas, args):
    """Yield completed shapes using the standard worker and benchmark.

    Keep scheduling here so only this adapter adds per-shape CSV output.
    The shared worker, reference checks, rotation and timer are unchanged.
    Task-start tracking also retains the shared scheduler's fault watchdog.
    """
    groups = {}
    for task in tasks:
        groups.setdefault(task[0][0], []).append(task)
    groups = list(groups.values())
    if not groups:
        return
    if len(groups) != len(in_datas):
        raise ValueError("Each shape must have exactly one input-data group")
    gpu_num = torch.cuda.device_count()
    if not gpu_num:
        raise RuntimeError("No GPUs available for tuning")
    parallel_num = gpu_num if args.mp < 1 else min(args.mp, gpu_num)
    ctx = mp.get_context("spawn")
    started = ctx.RawArray("d", len(groups))
    pids = ctx.RawArray("i", len(groups))
    pool = None

    def submit(indices):
        nonlocal pool
        pool = ctx.Pool(
            parallel_num, initializer=mp_workers._init_task_start_times,
            initargs=(started, pids),
        )
        workers = [pool.apply_async(mp_workers.get_pid) for _ in range(parallel_num)]
        gpu_map = {worker.get(): gpu for gpu, worker in enumerate(workers)}
        mp_workers._reset_task_start_times(started, indices, pids)
        pending = {
            index: pool.apply_async(
                mp_workers._run_with_start_tracking,
                args=(index, mp_workers.work_group, (
                    gpu_map, True, args.errRatio, in_datas[index],
                    groups[index], args.verbose,
                )),
            )
            for index in indices
        }
        pool.close()
        return pending

    def stop_pool():
        nonlocal pool
        if pool is not None:
            pool.terminate()
            pool.join()
            pool = None

    print(f"Tuning {len(groups)} shape groups on {parallel_num} GPUs", flush=True)
    try:
        pending = submit(list(range(len(groups))))
        while pending:
            made_progress, restart = False, False
            for index, future in list(pending.items()):
                reason = None
                if future.ready():
                    try:
                        results = future.get()
                    except Exception as exc:
                        reason = repr(exc)
                        restart = (
                            mp_workers._is_mapping_error(exc)
                            or mp_workers._is_accelerator_error(exc)
                        )
                        if mp_workers._is_mapping_error(exc):
                            break  # Refresh the GPU/PID map before resubmitting.
                else:
                    elapsed = mp_workers._elapsed_since_task_start(started, index)
                    if elapsed is None:
                        continue
                    if mp_workers._task_worker_exited(pids, index):
                        reason, restart = "worker exited", True
                    elif args.timeout is not None and elapsed > args.timeout:
                        reason, restart = f"timeout after {elapsed:.1f}s", True
                    else:
                        continue
                if reason is not None:
                    logger.error(f"Shape {groups[index][0][0][0]} failed: {reason}")
                    results = [(task[0], float("inf"), 1.0) for task in groups[index]]
                del pending[index]
                made_progress = True
                yield results
                if restart:
                    break
            if restart and pending:
                remaining = list(pending)
                stop_pool()
                pending = submit(remaining)
            elif pending and not made_progress:
                time.sleep(0.1)
    finally:
        stop_pool()


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
    m, k = x.shape
    n = w.shape[0]
    workspace = _tune_workspaces().get((x.device.index, m, n, k, kid))
    opus_gemm(
        x,
        w,
        out,
        kid=kid,
        layout="bpreshuffle",
        x_scale=x_scale,
        w_scale=w_scale,
        workspace=workspace,
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


_FLYDSL_BASELINE = str(
    Path(__file__).resolve().parents[2]
    / "aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
)
_FLYDSL_BMM_KEYS = ("x", "weight_shuffle", "x_scale_bmm", "w_scale_bmm", "out")
_OPUS_REF_KEYS = ("reference_bounds",)
_FLYDSL_REF_KEYS = ("reference_bf16",)


@functools.lru_cache(maxsize=8)
def load_flydsl_baseline(path):
    """Read only the gfx950 128x128 native-scale BMM catalog rows."""
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"gfx", "M", "N", "K", "libtype", "kernelId", "splitK", "kernelName"}
    if rows and required.difference(rows[0]):
        raise ValueError(f"FlyDSL baseline missing columns: {sorted(required.difference(rows[0]))}")
    return tuple(
        row for row in rows
        if row["gfx"] == "gfx950" and row["libtype"] == "flydsl"
        and row["kernelId"] == "bmm"
        and row.get("w_scale_block", "128x128") == "128x128"
    )


def validate_flydsl_candidate(kernel_id, name, split_k, m, n, k):
    """Validate the gfx950 native-scale BMM launch contract."""
    if kernel_id != "bmm":
        raise ValueError("This tuner only supports FlyDSL kernelId=bmm")
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import parse_bmm_kernel_name
    from aiter.ops.flydsl.kernels.bmm_a8w8_mxscale_gfx950 import check_bmm_config

    cfg = parse_bmm_kernel_name(name)
    if cfg is None or cfg["splits"] != split_k:
        raise ValueError(f"Invalid FlyDSL BMM name/splitK: {name!r}/{split_k}")
    check_bmm_config(
        n, k, 1, **cfg, x_scale_k=128, w_scale_n=128,
        w_scale_k=128, x_scale_transposed=m > 1,
    )
    tiles = math.ceil(m / cfg["tile_m"]) * (n // cfg["tile_n"])
    if cfg["splits"] > 1 and tiles > (1 << 16):
        raise ValueError("FlyDSL BMM split-K arrival-counter capacity exceeded")
    return cfg


def flydsl_bmm_candidates(m, n, k, baseline_path=_FLYDSL_BASELINE):
    """All legal baseline BMM names plus the latest per-shape heuristic.

    Upstream supplies no finite BMM tuning catalog. The baseline name union is
    an explicit, reproducible search space; it retains every saved baseline
    candidate and includes the current fallback for shapes without a saved row.
    """
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
        parse_bmm_kernel_name,
        pick_bmm_kernel_name,
    )

    rows = load_flydsl_baseline(str(baseline_path))
    names = {row["kernelName"] for row in rows if row["kernelId"] == "bmm"}
    names.add(pick_bmm_kernel_name(1, m, n, k, 128, 128, 128, m > 1))
    result = []
    for name in sorted(names):
        cfg = parse_bmm_kernel_name(name)
        if cfg is None:
            raise ValueError(f"Unrecognized FlyDSL baseline BMM name: {name!r}")
        try:
            validate_flydsl_candidate("bmm", name, cfg["splits"], m, n, k)
        except ValueError:
            continue
        result.append(("bmm", cfg["splits"], name))
    # A malformed or newly incompatible own-shape row must never disappear.
    for row in rows:
        if row["kernelId"] == "bmm" and (int(row["M"]), int(row["N"]), int(row["K"])) == (m, n, k):
            validate_flydsl_candidate("bmm", row["kernelName"], int(row["splitK"]), m, n, k)
    return result


def generate_native_tune_data(m, n, k, seed, device="cuda"):
    """Prepare native shared storage and a reference before timing kernels.

    A scales are logical [M,K/128] with physical [K/128,M] storage. BMM
    views the same bytes and sets x_scale_transposed=True in its runner.
    """
    data = generate_data(m, n, k, seed, device=device)
    data["weight_shuffle"] = data["w"]
    data["x_scale_bmm"] = data["x_scale"].T.view(m, k // 128)
    data["w_scale_bmm"] = data["w_scale"]
    data["reference_bounds"] = run_torch(
        data["x"], data["w_reference"], data["x_scale"], data["w_scale"],
        with_bounds=True,
    )
    data["reference_bf16"] = data["reference_bounds"][0].to(data["out"].dtype)
    workspaces = _tune_workspaces()
    device_index = data['x'].device.index
    for cache_key in list(workspaces):
        if cache_key[:4] != (device_index, m, n, k):
            del workspaces[cache_key]
    buffers = {}
    for kid in candidate_kids_for_shape('gfx950',m,n,k,exhaustive=True):
        instance = a8w8_mxscale_gemm_bpreshuffle_kernels_list[kid]
        size = bpreshuffle_workspace_bytes(instance,m,n,k)
        if size:
            key = (device_index,m,n,k,kid)
            if key not in workspaces:
                zero_init = bool(instance.bpreshuffle_entry and
                                 instance.bpreshuffle_entry.get('workspace_zero_init'))
                buffer_key = (size, kid if zero_init else None)
                if buffer_key not in buffers:
                    allocate = torch.zeros if zero_init else torch.empty
                    buffers[buffer_key] = allocate(
                        (size+3)//4,dtype=torch.float32,device=data['x'].device)
                workspaces[key] = buffers[buffer_key]
    return data


@functools.lru_cache(maxsize=1)
def _tune_workspaces():
    return {}


def get_flydsl_bmm_tune_tasks(info_keys, seed, run_kwargs, baseline_path=_FLYDSL_BASELINE):
    gfx, _, m, n, k = info_keys
    if gfx != "gfx950":
        return []
    return [
        ((info_keys, kid, sk, name, "flydsl", True), generate_native_tune_data,
         (m, n, k, seed), run_flydsl_bmm, (_FLYDSL_BMM_KEYS, name), dict(run_kwargs),
         reference_from_data, (_FLYDSL_REF_KEYS,), {}, None,
         1e-2, 0.01, None, None, ("out",))
        for kid, sk, name in flydsl_bmm_candidates(m, n, k, baseline_path)
    ]


def reference_from_data(reference):
    """Return the caller-prepared oracle; switching backend adds no matmul."""
    return reference


def run_flydsl_bmm(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import run_bmm_a8w8_mxfp8_gfx950

    m, k = x.shape
    n = out.shape[1]
    run_bmm_a8w8_mxfp8_gfx950(
        x.view(m, 1, k), weight_shuffle.view(1, n, k),
        x_scale.view(m, 1, k // 128), w_scale.view(1, n // 128, k // 128),
        out.view(m, 1, n), kernel_name=kernel_name, x_scale_transposed=True,
    )
    return out


class OpusMxscaleBpreshuffleTuner(generic_tune.GemmA8W8BlockScaleTuner):
    ARG_DEFAULTS: ClassVar[dict] = {
        **generic_tune.GemmA8W8BlockScaleTuner.ARG_DEFAULTS,
        "tune_file": "/tmp/opus_mxscale_bpreshuffle_tuned.csv",
        "untune_file": "",
        # Inherit the FlyDSL error-ratio threshold.
        # OPUS candidates retain their stricter zero-outlier contract.
        # Exact replay calls backend tuning entries directly. Publishing this
        # dataset as a production FP32-scale config would change its contract.
        "config_env_name": None,
    }

    def __init__(self):
        self.opus_all_candidates = False
        self._compare_backends = False
        self._sort_output = True
        self.opus_kids = None
        self.opus_pipelines = None
        self.opus_max_candidates = 10
        self.flydsl_baseline = _FLYDSL_BASELINE
        super().__init__(
            "opus_mxscale_bpreshuffle",
            # Input/output/scale dtypes are fixed. Keep them as validated
            # internal defaults, not serialized shape
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
                *CONFIG_COLUMNS,
            ],
            description=(
                "Tune gfx950 FP8 B-preshuffle GEMM with OPUS and FlyDSL BMM, "
                "BF16 output and shared native E8M0 scales (A 1x128, B 128x128). "
                "Measures backend GPU time."
            ),
        )
        self.backend_columns = list(self.columns)

    def _setup_specific_arguments(self):
        super()._setup_specific_arguments()
        self.parser.add_argument(
            "--opus-all-candidates", action="store_true",
            help="search every formal OPUS configuration registered for the shape",
        )
        self.parser.add_argument(
            "--opus-kids", "--opus_kids",
            type=parse_opus_kids,
            default=None,
            help="explicit registered OPUS IDs; default: prioritized parameter configurations; ignored for FlyDSL-only tuning and --run_config",
        )
        self.parser.add_argument(
            "--opus-pipelines", type=lambda value: tuple(value.split(",")), default=None,
            help="comma-separated configuration pipelines to explore; default: all supported pipelines",
        )
        self.parser.add_argument(
            "--opus-max-candidates", type=int, default=10,
            help="maximum prioritized configurations per shape (default: 10); explicit --opus-kids bypasses this limit",
        )
        self.parser.add_argument(
            "--flydsl-baseline", default=_FLYDSL_BASELINE,
            help="FlyDSL BMM catalog CSV: search legal saved gfx950 BMM names plus the shape heuristic",
        )
        # This adapter is B-preshuffle-only.  Keep the generic flag accepted so
        # existing tune commands remain valid, but make it true by default.
        self.parser.set_defaults(preshuffle=True, libtype="both")
        for action in self.parser._actions:
            if action.dest == "libtype":
                action.choices = ["both", "opus", "flydsl_bmm"]
                action.help = "both (default): tune and compare OPUS/FlyDSL per shape into one CSV; opus or flydsl_bmm: single backend"
            elif action.dest == "preshuffle":
                action.help = "accepted for compatibility; always enabled by this tuner"
            elif action.dest == "run_config":
                action.help = (
                    "replay saved exact backend/kernel choices from TUNED_CSV "
                    "(or -o when omitted), filtered to --libtype; measures backend GPU time"
                )
            elif action.dest == "splitK":
                action.help = (
                    "accepted for compatibility; OPUS uses splitK=0 and FlyDSL BMM "
                    "always enumerates the catalog's supported split-K configs"
                )
            elif action.dest == "blockPerCu":
                action.help = "accepted for compatibility; unused by OPUS and FlyDSL BMM"
            elif action.dest == "errRatio":
                action.help = (
                    "maximum outlier fraction for FlyDSL BMM "
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
        if args.libtype == "both":
            return self._run_both(args)
        return GemmCommonTuner.run(self, args, fast_mode)

    def _run_both(self, args):
        self.pre_process(args)
        output = self.get_out_file(args.tune_file)
        if self.untunedf.empty:
            logger.info(f"All requested shapes already compared in {output}")
            return self.tunedf
        if args.batch < 1:
            self.parser.error("--batch must be positive")
        self.tune_start_time = time.time()
        status = "Finished"
        try:
            for start in range(0, len(self.untunedf), args.batch):
                batch = self.untunedf.iloc[start:start + args.batch]
                tasks, inputs = self._build_tasks(batch, args)
                for results in tune_shape_groups(tasks, inputs, args):
                    row = self.post_process(results, args, topk=1)
                    self.result_to_csv(row, output)
                    best = row.iloc[0]
                    print(
                        f"[{len(self.success) + len(self.failed)}/{len(self.untunedf)}] "
                        f"M={best.M},N={best.N},K={best.K}: "
                        f"OPUS={best.opus_us}us, FlyDSL={best.flydsl_us}us, "
                        f"winner={best.winner}, BMM/OPUS={best.BMM_over_OPUS}; "
                        f"{best.comparison_status}; saved {output}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            status = "Interrupted"
        except Exception:
            status = "Error"
            raise
        finally:
            # Each completed shape is already persisted. Summary adds no timing.
            if status != "Error":
                self.tune_summary(status)
        return self.tunedf

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
            *(("splitK",) if saved else ()),
        ):
            value = pd.to_numeric(df[column], errors="raise")
            minimum = 0 if column in ("kernelId", "splitK") else 1
            if (value.isna() | (value < minimum) | (value % 1 != 0)).any():
                raise ValueError(f"{column} must contain integers >= {minimum}")
            df[column] = value.astype("int64")
        if saved:
            is_bmm = df.libtype.eq("flydsl") & df.kernelId.astype(str).eq("bmm")
            numeric = ~is_bmm
            numeric_ids = pd.to_numeric(df.loc[numeric, "kernelId"], errors="raise")
            if (numeric_ids.isna() | (numeric_ids < 0) | (numeric_ids % 1 != 0)).any():
                raise ValueError("kernelId must be a nonnegative integer or FlyDSL 'bmm'")
            df["kernelId"] = df["kernelId"].astype(object)
            df.loc[numeric, "kernelId"] = numeric_ids.astype("int64")
            df.loc[is_bmm, "kernelId"] = "bmm"
        if "w_scale_block" in df and not df.w_scale_block.eq("128x128").all():
            raise ValueError("MXFP8 bpreshuffle adapter requires w_scale_block=128x128")
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
        # Saved rows from a mixed CSV remain readable for mode filtering.
        if saved:
            df["scale_dtype"] = df["libtype"].isin(_SUPPORTED_LIBTYPES).map(
                {True: "e8m0", False: "fp32"}
            )
        return df

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        # Read the complete schema directly: the production MXPSH tuner hides
        # BMM rows while retuning, whereas this adapter selects them for replay.
        df = GemmCommonTuner.get_tuned_gemm_list(self, tuned_gemm_file, columns)
        if df.empty:
            return df
        if self._compare_backends:
            missing = set(_COMPARISON_COLUMNS).difference(df.columns)
            if missing:
                raise ValueError("--libtype both requires a combined -o CSV; choose a new output path")
        return self._with_configuration_metadata(
            self._normalize_rows(df, saved=True), validate=True,
        ).reindex(columns=self.columns)

    def _with_configuration_metadata(self, df, *, validate=False):
        """Populate fresh results, or validate complete saved configurations."""
        df = df.copy()
        for column in CONFIG_COLUMNS:
            if validate and column not in df:
                raise ValueError(f"Saved tuning CSV is missing {column}; rerun tuning")
            df[column] = df[column].astype(object) if validate else None
        for index, row in df.iterrows():
            if row.libtype == "opus":
                expected = configuration_metadata(
                    int(row.kernelId), int(row.M), int(row.N), int(row.K)
                )
            else:
                continue
            for column, value in expected.items():
                if validate:
                    saved = row[column]
                    if value is None:
                        # The shared writer uses na_rep="Null", which pandas
                        # does not recognize as a missing value on CSV reload.
                        matches = pd.isna(saved) or saved == "Null"
                    elif column == "configJson":
                        matches = pd.notna(saved) and json.loads(str(saved)) == json.loads(value)
                    elif isinstance(value, bool):
                        matches = pd.notna(saved) and saved in (value, str(value))
                    elif isinstance(value, int):
                        # An axis containing both numbers and "Null" reloads
                        # as strings. Compare its numeric value without
                        # accepting missing or fractional configuration data.
                        matches = pd.to_numeric(saved, errors="coerce") == value
                    else:
                        matches = pd.notna(saved) and saved == value
                    if not matches:
                        raise ValueError(
                            f"Saved OPUS {column} disagrees with kernelId={row.kernelId} "
                            f"for shape {(row.M, row.N, row.K)}"
                        )
                df.at[index, column] = value
        return df

    def pre_process(self, args):
        self._compare_backends = args.libtype == "both" and not args.run_config
        self._sort_output = args.sort
        self.columns = self.backend_columns + (list(_COMPARISON_COLUMNS) if self._compare_backends else [])
        self.success = pd.DataFrame(columns=self.columns)
        self.failed = pd.DataFrame(columns=self.columns)
        self.opus_all_candidates = args.opus_all_candidates
        gfx, cu_num = self.get_gfx(), self.get_cu_num()
        if gfx != "gfx950":
            self.parser.error(
                f"MXFP8 bpreshuffle tuning/replay only support gfx950; got {gfx}"
            )
        args.preshuffle = True
        self.opus_kids = None
        self.opus_pipelines = getattr(args, "opus_pipelines", None)
        self.opus_max_candidates = getattr(args, "opus_max_candidates", 10)
        if self.opus_max_candidates <= 0:
            self.parser.error("--opus-max-candidates must be positive")
        self.flydsl_baseline = args.flydsl_baseline
        if args.libtype not in ("both", "opus", "flydsl_bmm"):
            self.parser.error("--libtype must be both, opus or flydsl_bmm")
        if args.libtype in ("both", "flydsl_bmm") and not args.run_config and not self.opus_all_candidates:
            load_flydsl_baseline(str(self.flydsl_baseline))
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
            if args.libtype == "flydsl_bmm":
                self.untunedf = self.untunedf[
                    self.untunedf.libtype.eq("flydsl")
                    & self.untunedf.kernelId.astype(str).eq("bmm")
                ].reset_index(drop=True)
            elif args.libtype == "opus":
                self.untunedf = self.untunedf[
                    self.untunedf.libtype.isin(_OPUS_LIBTYPES)
                ].reset_index(drop=True)
            if self.untunedf.empty:
                raise ValueError(f"No saved rows match --libtype {args.libtype}")
            return
        if not args.untune_file:
            self.parser.error("--input_file/-i is required")
        self.opus_kids = (
            getattr(args, "opus_kids", None)
            if args.libtype in ("both", "opus")
            else None
        )
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
        if args.libtype in ("both", "opus"):
            for row in self.untunedf.itertuples(index=False):
                if not self._candidate_kids(row.gfx, row.M, row.N, row.K):
                    raise ValueError(
                        f"No MXFP8 bpreshuffle candidate supports "
                        f"{(row.M, row.N, row.K)}"
                    )
        self.tunedf = self.get_tuned_gemm_list(args.tune_file)
        if not self.tunedf.empty and args.libtype in ("opus", "flydsl_bmm"):
            selected = self.tunedf.libtype.isin(_OPUS_LIBTYPES) if args.libtype == "opus" else (
                self.tunedf.libtype.eq("flydsl")
                & self.tunedf.kernelId.astype(str).eq("bmm")
            )
            if not selected.all():
                self.parser.error(
                    f"--libtype {args.libtype} requires its own -o result CSV; "
                    "the existing output contains other backend rows"
                )
        if not args.all and not self.tunedf.empty:
            completed = self.tunedf
            if self._compare_backends:
                completed = completed[completed.comparison_status.eq("ok")]
            else:
                completed = completed[pd.to_numeric(completed.us, errors="coerce").gt(0)]
            self.untunedf = self.untunedf[
                ~self.untunedf.apply(tuple, axis=1).isin(
                    completed[self.keys].apply(tuple, axis=1)
                )
            ].reset_index(drop=True)

    def _candidate_kids(self, gfx, m, n, k):
        kids = candidate_kids_for_shape(
            gfx, m, n, k, all_candidates=self.opus_kids is not None,
            max_candidates=self.opus_max_candidates,
            pipelines=self.opus_pipelines if self.opus_kids is None else None,
            cu_num=self.get_cu_num(), exhaustive=self.opus_all_candidates,
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
        gen_data=generate_native_tune_data,
        ref_keys=_OPUS_REF_KEYS,
        gen_extra_args=(),
        compare_fn=compare_outputs,
    ):
        return (
            info,
            gen_data,
            (m, n, k, seed, *gen_extra_args),
            func,
            (arg_keys, *extra_args),
            dict(run_kwargs),
            reference_from_data,
            (ref_keys,),
            {},
            None,
            1e-2,
            1e-2,
            compare_fn,
            None,
            ("out",),
        )

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
            tasks.append(self._make_task(
                info, m, n, k, seed, run_bench, _BENCH_KEYS, (kid,), run_kwargs,
            ))
        return tasks

    def _build_tasks(self, untunedf, args):
        if args.libtype not in ("both", "opus", "flydsl_bmm"):
            raise ValueError("--libtype must be both, opus or flydsl_bmm")
        self.opus_kids = (
            getattr(args, "opus_kids", None)
            if args.libtype in ("both", "opus")
            else None
        )
        requested_kids = set()
        if args.libtype in ("both", "opus"):
            for row in untunedf.itertuples(index=False):
                kids = self._candidate_kids(row.gfx, row.M, row.N, row.K)
                if not kids:
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
            if args.libtype in ("both", "opus"):
                tasks.extend(self.get_gemm_a8w8_blockscale_opus_tune_task(
                    info, 0, True, run_kwargs
                ))
            if args.libtype in ("both", "flydsl_bmm"):
                if self.opus_all_candidates:
                    for name, split in bpreshuffle_bmm_candidates(row.M, row.N, row.K):
                        validate_flydsl_candidate("bmm", name, split, row.M, row.N, row.K)
                        tasks.append(self._make_task(
                            (info, "bmm", split, name, "flydsl", True), row.M, row.N, row.K, 0,
                            run_flydsl_bmm, _FLYDSL_BMM_KEYS, (name,), run_kwargs,
                            ref_keys=_FLYDSL_REF_KEYS, compare_fn=None,
                        ))
                else:
                    tasks.extend(get_flydsl_bmm_tune_tasks(
                        info, 0, run_kwargs, self.flydsl_baseline
                    ))
            if args.libtype == "both":
                backends = {task[0][4] for task in tasks[start:]}
                if not backends.intersection(_OPUS_LIBTYPES) or "flydsl" not in backends:
                    raise ValueError(f"Both backends need legal candidates for {info}")
            if len(tasks) != start:
                tasks_data.append((len(tasks) - start, ()))
        return tasks, tasks_data

    def tune(self, untunedf, tunedf, args):
        tasks, tasks_data = self._build_tasks(untunedf, args)
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
        return 0.0 if libtype in _OPUS_LIBTYPES else args.errRatio

    def _comparison_results(self, rets, args):
        grouped = {}
        for result in rets:
            grouped.setdefault(result[0][0], []).append(result)
        rows = []
        for shape, results in grouped.items():
            best = {}
            for result in results:
                info, us, error = result
                if not (math.isfinite(us) and us > 0 and math.isfinite(error)
                        and 0 <= error <= self._error_limit(info[4], args)):
                    continue
                backend = "opus" if info[4] in _OPUS_LIBTYPES else "flydsl"
                if backend not in best or us < best[backend][1]:
                    best[backend] = result
            winner = min(best, key=lambda backend: best[backend][1]) if best else None
            selected = best[winner] if winner else (results[0][0], self.INVALID_TIME, 1.0)
            row = self.result_to_df([selected]).iloc[0].to_dict()
            row.update(dict.fromkeys(_COMPARISON_COLUMNS))
            for backend, result in best.items():
                saved = self.result_to_df([result]).iloc[0]
                for column in ("kernelId", "splitK", "us", "kernelName", "errRatio"):
                    row[backend + "_" + column] = saved[column]
                if backend == "opus":
                    row["opus_backend"] = saved.libtype
                    for column in CONFIG_COLUMNS:
                        row["opus_" + column] = saved[column]
            if len(best) == 2:
                ratio = best["flydsl"][1] / best["opus"][1]
                row.update(
                    winner="tie" if ratio == 1 else winner,
                    BMM_over_OPUS=ratio, OPUS_speedup_pct=(ratio - 1) * 100,
                    comparison_status="ok",
                )
            else:
                missing = [backend for backend in ("opus", "flydsl") if backend not in best]
                row.update(winner="unavailable", comparison_status="no_valid_" + "_and_".join(missing))
                logger.error(f"Incomplete comparison for {shape}: {row['comparison_status']}")
            rows.append(row)
        return pd.DataFrame(rows, columns=self.columns)

    def result_to_csv(self, resultdf, file, concat=False):
        if not self._compare_backends:
            return GemmCommonTuner.result_to_csv(self, resultdf, file, concat)
        complete = resultdf.comparison_status.eq("ok")
        self.success = pd.concat([self.success, resultdf[complete]], ignore_index=True)
        self.failed = pd.concat([self.failed, resultdf[~complete]], ignore_index=True)
        # One parent process writes each completed shape, including failures so
        # resume can retry them. Retuning replaces the row, never an unrelated shape.
        old = self.tunedf
        keys = resultdf[self.keys].apply(tuple, axis=1)
        old = old[~old[self.keys].apply(tuple, axis=1).isin(keys)]
        updated = pd.concat([old, resultdf], ignore_index=True)
        updated = updated.drop_duplicates(subset=self.keys, keep="last")
        if self._sort_output:
            updated = updated.sort_values(self.sort_keys)
        output = Path(file)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            updated.to_csv(temporary, index=False, na_rep="Null")
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        self.tunedf = updated.reset_index(drop=True)

    def post_process(self, rets, args, topk=-1, fast_mode=False):
        rets = list(rets)
        # Save the original timings and measured error ratios without alteration.
        # The shared profile sort compares integer IDs with the string "bmm".
        # Use a stable string key here; selection below sorts only by timings.
        if args.profile_file:
            ordered = sorted(
                rets, key=lambda result: (
                    result[0][0], str(result[0][4]),
                    str(result[0][1]), result[0][2], str(result[0][3]),
                ),
            )
            profile = self.result_to_df(ordered)
            profile_path = Path(args.profile_file)
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            if self._compare_backends:
                exists = profile_path.is_file() and profile_path.stat().st_size > 0
                profile[self.backend_columns].to_csv(
                    profile_path, mode="a", header=not exists, index=False, na_rep="Null",
                )
            else:
                if profile_path.is_file():
                    profile = pd.concat([pd.read_csv(profile_path), profile], ignore_index=True)
                profile.to_csv(profile_path, index=False, na_rep="Null")
        raw = rets
        if fast_mode or topk == -1:
            return raw
        if self._compare_backends:
            return self._comparison_results(rets, args)
        # The shared selector has one threshold. Mark rejected candidates as
        # unavailable for selection while retaining their raw profile records.
        selection = [
            (
                info,
                us if math.isfinite(us) and us > 0 and math.isfinite(error)
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
        # BMM names carry their launch parameters and are supplied by tasks.
        return None

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
        frame = super().result_to_df(normalized)
        # Failed BMM candidates share kernelId="bmm"; their task name is the
        # only complete configuration identity and must survive serialization.
        for index, (info, _, _) in enumerate(normalized):
            if info[3]:
                frame.at[index, "kernelName"] = info[3]
        return self._with_configuration_metadata(frame).reindex(columns=self.backend_columns)

    def run_config(self, args):
        from aiter.test_common import checkAllclose, run_perftest

        results = []
        rows = self._with_configuration_metadata(
            self._normalize_rows(self.untunedf, saved=True), validate=True,
        )
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
                    row.gfx, row.M, row.N, row.K, row.outdtype, all_candidates=True,
                )
            ):
                raise ValueError(
                    f"Saved OPUS kid {row.kernelId} is incompatible with {row}"
                )
            saved_name = getattr(row, "kernelName", None)
            if row.libtype == "flydsl":
                if pd.isna(saved_name) or not str(saved_name).strip():
                    raise ValueError("Saved FlyDSL row requires kernelName")
                validate_flydsl_candidate(
                    row.kernelId, str(saved_name), row.splitK, row.M, row.N, row.K,
                )
            elif pd.notna(saved_name) and str(saved_name).strip():
                expected_name = self.getKernelName(row.kernelId, row.libtype, True)
                if saved_name != expected_name:
                    raise ValueError(
                        f"Saved {row.libtype} kernelName does not match kid {row.kernelId}"
                    )
        if not opus_rows.empty:
            _ensure_kids_compiled(set(opus_rows.kernelId))
        native_data_key, native_data = None, None
        for row in rows.itertuples(index=False):
            kid = row.kernelId
            ref_keys = _OPUS_REF_KEYS if row.libtype in _OPUS_LIBTYPES else _FLYDSL_REF_KEYS
            key = (row.M, row.N, row.K)
            if key != native_data_key:
                native_data = generate_native_tune_data(*key, 0, device="cuda")
                native_data_key = key
            data = native_data
            ref = reference_from_data(*(data[key] for key in ref_keys))
            data["out"].fill_(float("nan"))
            if row.libtype == "opus":
                bench, bench_args = run_bench, (
                    *(data[key] for key in _BENCH_KEYS), kid,
                )
            else:
                bench = run_flydsl_bmm
                bench_args = (*(data[key] for key in _FLYDSL_BMM_KEYS), row.kernelName)
            out, us = run_perftest(
                bench,
                *bench_args,
                num_warmup=args.warmup,
                num_iters=args.iters,
            )
            if row.libtype in _OPUS_LIBTYPES:
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
