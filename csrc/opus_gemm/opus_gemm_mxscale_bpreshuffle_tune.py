# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gfx950 MXFP8 B-preshuffle GEMM across OPUS and common backends.

This gfx950 adapter keeps each backend's scale input format: OPUS and both
FlyDSL paths receive the same random native E8M0 values, while CK, CKTile and
ASM reuse the original blockscale tuner's random FP32 scales, reference and
accuracy checks.
"""

import argparse
import itertools
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
    A8W8_BPRESHUFFLE_FAMILIES,
    a8w8_mxscale_bpreshuffle_candidate_kids,
    A8W8_BPRESHUFFLE_TUNING_KIDS,
    a8w8_mxscale_bpreshuffle_supports_shape,
    a8w8_mxscale_gemm_bpreshuffle_kernels_list,
    bpreshuffle_candidate_split_k,
    bpreshuffle_launch_plan,
    canonical_output_dtype,
)
from csrc.opus_gemm.opus_gemm_bpreshuffle_config import (
    BPRESHUFFLE_PIPELINE_NAMES,
    BpreshuffleConfig,
    config_identity,
    config_from_legacy_kid,
    construct_config,
    config_supports_shape,
    kernel_instance_from_config,
    pipeline_configs,
    validate_saved_config,
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
_NATIVE_LIBTYPES = frozenset(("opus", "flydsl", "flydsl_bmm"))
_SUPPORTED_LIBTYPES = frozenset(("ck", "cktile", "asm", *_NATIVE_LIBTYPES))
_FLYDSL_BMM_KEYS = ("x_bmm", "w_bmm", "out_bmm", "x_scale_bmm", "w_scale_bmm")
_FLYDSL_KEYS = ("x", "w", "out", "x_scale_shuf", "w_scale_shuf")


def selected_backends(libtype):
    if libtype == "all":
        return _SUPPORTED_LIBTYPES
    if libtype == "both":
        return frozenset(("ck", "cktile"))
    if libtype == "opus_flydsl":
        return _NATIVE_LIBTYPES
    if libtype not in _SUPPORTED_LIBTYPES:
        raise ValueError(f"Unsupported backend selection: {libtype}")
    return frozenset((libtype,))


def flydsl_modules(libtype):
    """Load the selected runtime explicitly; never silently omit FlyDSL."""
    try:
        if libtype == "flydsl_bmm":
            from aiter.ops.flydsl import batched_gemm_a8w8_gfx950 as runtime
            return runtime
        from aiter.ops.flydsl import mxscale_preshuffle_kernels as runtime
        from aiter.ops.flydsl.gemm_tune import flydsl_gemm_mxscale_preshuffle_common as catalog
        # The host dispatcher imports the compiler only on first launch. Load
        # its kernel here so a stale installation fails before other builds.
        from aiter.ops.flydsl.kernels.gemm import mxscale_preshuffle  # noqa: F401
        return runtime, catalog
    except ImportError as exc:
        raise RuntimeError(
            f"{libtype} requires a compatible FlyDSL installation (the validated "
            "environment uses 0.3.4.1). Select another --libtype or fix PYTHONPATH."
        ) from exc


def parse_flydsl_kids(value):
    try:
        return parse_opus_kids(value)
    except argparse.ArgumentTypeError as exc:
        raise argparse.ArgumentTypeError(str(exc).replace("--opus-kids", "--flydsl-kids")) from exc


def parse_flydsl_bmm_kernels(value):
    names = tuple(item.strip() for item in value.split(","))
    if not all(names) or len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("--flydsl-bmm-kernels requires unique comma-separated kernel names")
    return names


def validate_flydsl_bmm(runtime, name, m, n, k, split_k=None):
    config = runtime.parse_bmm_kernel_name(name)
    if config is None or min(m, n, k) <= 0 or n % 128 or k % 128:
        raise ValueError(f"Invalid FlyDSL BMM name/shape: {name}, {(m, n, k)}")
    if any(config[key] <= 0 for key in (
        "tile_m", "tile_n", "tile_k", "m_warp", "n_warp", "num_buffers", "splits"
    )):
        raise ValueError(f"FlyDSL BMM parameters must be positive: {name}")
    if runtime.bmm_kernel_name(**config) != name:
        raise ValueError(f"Noncanonical FlyDSL BMM kernelName: {name}")
    if split_k is not None and split_k != config["splits"]:
        raise ValueError("Saved FlyDSL BMM splitK does not match kernelName")
    runtime.check_bmm_config(
        n, k, 1, **config, x_scale_k=128, w_scale_n=128,
        w_scale_k=128, x_scale_transposed=m > 1,
    )
    tiles = math.ceil(m / config["tile_m"]) * (n // config["tile_n"])
    if config["splits"] > 1 and tiles > 65536:
        raise ValueError("FlyDSL BMM split-K exceeds 65536 arrival counters")
    return config


def flydsl_bmm_candidate_names(runtime, m, n, k, cu_num, names=None, search="balanced"):
    """Search BMM launch parameters, independently of the heuristic winner.

    This is an explicit local search space, not a reconstruction of upstream
    tuning history. Exact --flydsl-bmm-kernels subsets bypass search pruning.
    The official legality checker owns LDS, scale and cooperative-copy rules.
    """
    if names is not None:
        for name in names:
            validate_flydsl_bmm(runtime, name, m, n, k)
        return tuple(names)
    if n % 128 or k % 128:
        return ()
    if search not in ("balanced", "full"):
        raise ValueError(f"Unknown FlyDSL BMM search: {search}")
    candidates = {}
    try:
        name = runtime.pick_bmm_kernel_name(1, m, n, k, 128, 128, 128, m > 1)
        validate_flydsl_bmm(runtime, name, m, n, k)
        candidates[name] = None
    except ValueError:
        pass
    full = search == "full"
    tile_ms = ((16, 32, 48, 64, 96, 112, 128, 144, 192, 224, 256, 288) if full
               else (16, 32, 64, 96, 128, 192, 256))
    tile_ns = (16, 32, 64, 128, 256, 384, 512) if full else (32, 64, 128, 256)
    waves = (((1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2)) if full
             else ((1, 1), (1, 4), (2, 2), (4, 1)))
    stages = range(2, 13) if full else (2, 3, 4)
    for tm, tn, tk, (mw, nw), buffers in itertools.product(
        tile_ms, tile_ns, (128, 256, 512), waves, stages,
    ):
        if tm > max(64, math.ceil(m / 16) * 16) or n % tn or k % tk:
            continue
        if tm % (mw * 16) or tn % (nw * 16):
            continue
        tiles = math.ceil(m / tm) * (n // tn)
        for splits, preload, ahead in itertools.product(
            (1, 2, 4, 8, 16, 32), (True, False), range(min(buffers, 5 if full else 3)),
        ):
            if k % (tk * splits) or k // (tk * splits) < buffers - 1:
                continue
            if splits > 1 and (tiles > 65536 or tiles >= cu_num or tiles * splits > 2 * cu_num):
                continue
            cfg = dict(tile_m=tm, tile_n=tn, tile_k=tk, m_warp=mw, n_warp=nw,
                       num_buffers=buffers, splits=splits, b_direct=ahead > 0,
                       b_ahead=max(1, ahead), scale_preload=preload)
            try:
                runtime.check_bmm_config(n, k, 1, **cfg, x_scale_k=128,
                                        w_scale_n=128, w_scale_k=128, x_scale_transposed=m > 1)
            except ValueError:
                continue
            # These two knobs do not change the checker's resource constraints.
            for nt, xcd in itertools.product((False, True), repeat=2):
                name = runtime.bmm_kernel_name(**cfg, b_nt=nt, xcd_order=xcd)
                candidates[name] = None
    return tuple(candidates)


def prepare_flydsl_data(data, m, n, k):
    """Prepare descriptors/shuffled scales once, outside timed launches."""
    from aiter.ops.shuffle import shuffle_scale_blockscale_a, shuffle_scale_blockscale_b

    # A-scale is logical [M,K/128], physically [K/128,M]. The BMM descriptor
    # must view the physical bytes, while mxpsh shuffles the logical rows.
    descriptor = data["x_scale"].T.view(m, 1, k // 128)
    if not descriptor.is_contiguous() or descriptor.data_ptr() != data["x_scale"].data_ptr():
        raise ValueError("FlyDSL BMM activation-scale descriptor must preserve native storage")
    data.update(
        x_bmm=data["x"].view(m, 1, k), w_bmm=data["w"].view(1, n, k),
        out_bmm=data["out"].view(m, 1, n), x_scale_bmm=descriptor,
        w_scale_bmm=data["w_scale"].view(1, n // 128, k // 128),
        x_scale_shuf=shuffle_scale_blockscale_a(data["x_scale"].contiguous(), k),
        w_scale_shuf=shuffle_scale_blockscale_b(data["w_scale"], n, k),
    )
    return data


def run_flydsl_bmm_bench(x, w, out, x_scale, w_scale, name):
    runtime = flydsl_modules("flydsl_bmm")
    runtime.run_bmm_a8w8_mxfp8_gfx950(
        x, w, x_scale, w_scale, out, kernel_name=name, x_scale_transposed=True,
    )
    return out.view(x.shape[0], w.shape[1])


def run_flydsl_bench(x, w, out, x_scale, w_scale, name):
    runtime, _ = flydsl_modules("flydsl")
    return runtime.run_gemm_a8w8_mxscale_preshuffle_gfx950(x, w, x_scale, w_scale, out, name)


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


def parse_opus_pipelines(value):
    """Select a small set of pipeline bodies; tiles remain compile parameters."""
    names = [item.strip() for item in value.split(",")]
    unknown = set(names).difference(BPRESHUFFLE_PIPELINE_NAMES)
    if unknown or not names or len(names) != len(set(names)):
        raise argparse.ArgumentTypeError(
            "--opus_pipelines requires unique names from "
            + ",".join(sorted(BPRESHUFFLE_PIPELINE_NAMES))
        )
    return frozenset(names)


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


def candidate_kids_for_shape(gfx, m, n, k, outdtype="bf16", *, include_legacy=False, families=None):
    """Use registry tiling and byte limits before allocating tuning inputs."""
    return a8w8_mxscale_bpreshuffle_candidate_kids(
        gfx, m, n, k, outdtype, include_legacy=include_legacy, families=families,
    )


def generate_data(m, n, k, seed=0, flydsl=False, *, device):
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
    data = {
        "x": x,
        "w": weight_shuffle,
        "w_reference": weight,
        "out": torch.empty((m, n), device=device, dtype=torch.bfloat16),
        "x_scale": x_scale,
        "w_scale": w_scale,
    }
    return prepare_flydsl_data(data, m, n, k) if flydsl and n % 128 == 0 else data


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


def run_bench(x, w, out, x_scale, w_scale, kid, split_k=0):
    opus_gemm(
        x,
        w,
        out,
        kid=kid,
        layout="bpreshuffle",
        x_scale=x_scale,
        w_scale=w_scale,
        split_k=split_k,
    )
    return out


def run_config_bench(x, w, out, x_scale, w_scale, config, split_k=0):
    """Benchmark a canonical configuration through its independent JIT module."""
    from aiter.ops.opus import opus_gemm_bpreshuffle
    opus_gemm_bpreshuffle(
        x, w, out, x_scale, w_scale, pipeline=config.pipeline,
        config=config, split_k=split_k,
    )
    return out


def load_opus_configs(path):
    """Read additional compile tuples without registering numeric kernel IDs."""
    import json
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list) or not payload:
        raise ValueError("--opus_configs requires a non-empty JSON array")
    configs = []
    for row in payload:
        required = {"pipeline", "compile_params"}
        if (not isinstance(row, dict) or not required.issubset(row)
                or set(row).difference(required | {"legacy_kid", "identity"})):
            raise ValueError("Each OPUS config needs pipeline and compile_params; optional metadata: legacy_kid, identity")
        if not isinstance(row["compile_params"], dict):
            raise ValueError("compile_params must be a JSON object")
        config = construct_config(row["pipeline"], **row["compile_params"])
        if "legacy_kid" in row:
            if type(row["legacy_kid"]) is not int or row["legacy_kid"] < -1:
                raise ValueError("legacy_kid metadata must be an integer >= -1")
            validate_saved_config(row["pipeline"], row["compile_params"], row["legacy_kid"])
        if "identity" in row and row["identity"] != config_identity(config):
            raise ValueError("OPUS identity metadata does not match compile_params")
        configs.append(config)
    return tuple({(config.pipeline, config.to_json()): config for config in configs}.values())


def prepare_opus_configs(configs):
    """Compile configurations before timing; imports no tuner-side GPU probe."""
    from aiter.ops.opus.bpreshuffle_runtime import ensure_config
    for config in configs:
        ensure_config(config)


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


def compare_flydsl_outputs(ref, out, **kwargs):
    """Keep FlyDSL's BF16-reference/1% tolerance with shared native inputs."""
    from aiter.test_common import checkAllclose
    return checkAllclose(
        ref[0].to(out.dtype), out, rtol=1e-2, atol=1e-2,
        catastrophic_check=True, **kwargs,
    )


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
        self.opus_families = None
        self.opus_pipelines = None
        self.opus_configs = ()
        self.flydsl_kids = None
        self.flydsl_bmm_kernels = None
        self.flydsl_bmm_search = "balanced"
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
                "pipeline",
                "config",
            ],
            description=(
                "Tune gfx950 FP8 B-preshuffle GEMM across CK, CKTile, ASM, "
                "OPUS, FlyDSL BMM and FlyDSL numeric-ID mxpsh with BF16 output. "
                "CK/CKTile/ASM use random FP32 scales; OPUS/FlyDSL share "
                "random native E8M0 values. Measures backend GPU "
                "time, including internal transforms."
            ),
        )

    def _setup_specific_arguments(self):
        super()._setup_specific_arguments()
        self.parser.add_argument(
            "--opus-kids", "--opus_kids",
            type=parse_opus_kids,
            default=None,
            help=("legacy internal OPUS ABI ID subset; default: parameters of five pipelines filtered by shape. "
                  "Runtime IDs search all legal literal split counts; ignored for --run_config"),
        )
        self.parser.add_argument(
            "--opus-families", "--opus_families",
            default=None,
            help="legacy OPUS family filter; available: "
                 + ",".join(sorted(A8W8_BPRESHUFFLE_FAMILIES)) + "; default: all 89 active configurations",
        )
        self.parser.add_argument(
            "--opus_pipelines", "--opus-pipelines",
            type=parse_opus_pipelines,
            default=None,
            help="comma-separated pipelines to tune: "
                 + ",".join(sorted(BPRESHUFFLE_PIPELINE_NAMES))
                 + "; each pipeline enumerates its supported tile/wave/stage configurations; ignored for --run_config",
        )
        self.parser.add_argument(
            "--opus_configs", "--opus-configs", default=None,
            help="JSON array of additional {pipeline, compile_params} tuples to tune; no ID registration required",
        )
        self.parser.add_argument(
            "--flydsl-kids", type=parse_flydsl_kids, default=None,
            help="numeric-ID mxpsh candidate subset; default: full FP8 catalog; ignored for --run_config",
        )
        self.parser.add_argument(
            "--flydsl-bmm-kernels", type=parse_flydsl_bmm_kernels, default=None,
            help="exact comma-separated BMM kernelName subset; default: parameter search; ignored for --run_config",
        )
        self.parser.add_argument(
            "--flydsl-bmm-search", choices=("balanced", "full"), default="balanced",
            help="BMM parameter search breadth; full adds irregular tiles, up to 12 buffers and B-ahead=4",
        )
        # This adapter is B-preshuffle-only.  Keep the generic flag accepted so
        # existing tune commands remain valid, but make it true by default.
        self.parser.set_defaults(preshuffle=True)
        for action in self.parser._actions:
            if action.dest == "libtype":
                action.choices = ["ck", "cktile", "asm", "opus", "flydsl", "flydsl_bmm",
                                  "opus_flydsl", "both", "all"]
                action.help = (
                    "backend candidates: ck, cktile, asm, opus, flydsl (numeric-ID mxpsh), "
                    "flydsl_bmm, opus_flydsl (all three native E8M0 paths), "
                    "both (CK + CKTile), or all (all six backends)"
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
                    "include supported ASM split-K candidates; CK/CKTile use splitK=0. "
                    "OPUS/FlyDSL always search legal splits; fixed OPUS IDs use zero"
                )
            elif action.dest == "errRatio":
                action.help = (
                    "maximum outlier fraction for CK/CKTile/ASM/FlyDSL "
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
            if "kernelId" not in df and {"pipeline", "config"} <= set(df.columns):
                # Parameter-only rows do not require a compatibility ID.
                df["kernelId"] = -1
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
            # The upstream mxscale table uses flydsl/bmm for the BMM route.
            # Keep its sentinel, but give the two launch contracts distinct labels.
            bmm = df["kernelId"].astype(str).eq("bmm")
            if (bmm & ~df["libtype"].isin(("flydsl", "flydsl_bmm"))).any():
                raise ValueError("kernelId=bmm is only valid for flydsl_bmm rows")
            df.loc[bmm & df["libtype"].eq("flydsl"), "libtype"] = "flydsl_bmm"
            if (df["libtype"].eq("flydsl_bmm") & ~bmm).any():
                raise ValueError("flydsl_bmm rows require kernelId=bmm")
            numeric = pd.to_numeric(df.loc[~bmm, "kernelId"], errors="raise")
            if (numeric.isna() | (numeric < -1) | (numeric % 1 != 0)).any():
                raise ValueError("kernelId must contain integers >= -1 or the bmm sentinel")
            df["kernelId"] = df["kernelId"].astype(object)
            df.loc[~bmm, "kernelId"] = numeric.astype("int64")
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
            minimum = 0 if column == "splitK" else 1
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
            df["scale_dtype"] = df["libtype"].isin(_NATIVE_LIBTYPES).map(
                {True: "e8m0", False: "fp32"}
            )
            df = self._config_columns(df, validate_saved=True)
        return df

    @staticmethod
    def _config_columns(df, *, validate_saved=False):
        """Fill legacy metadata or validate new rows before any compilation."""
        df = df.copy()
        for column in ("pipeline", "config"):
            if column not in df:
                df[column] = ""
        pipelines, configs = [], []
        for row in df.itertuples(index=False):
            pipeline = str(row.pipeline).strip() if pd.notna(row.pipeline) else ""
            payload = str(row.config).strip() if pd.notna(row.config) else ""
            if row.libtype == "opus":
                if validate_saved and (pipeline or payload):
                    if not pipeline or not payload:
                        raise ValueError("Saved OPUS rows require both pipeline and config when either is present")
                    config = validate_saved_config(pipeline, payload, int(row.kernelId))
                elif pipeline and payload:
                    config = validate_saved_config(pipeline, payload, int(row.kernelId))
                else:
                    config = config_from_legacy_kid(int(row.kernelId))
                pipelines.append(config.pipeline)
                configs.append(config.to_json())
            else:
                if row.libtype not in _SUPPORTED_LIBTYPES:
                    raise ValueError(f"Unsupported saved libtype: {row.libtype}")
                if row.libtype == "flydsl_bmm":
                    if row.kernelId != "bmm":
                        raise ValueError("flydsl_bmm rows require kernelId=bmm")
                elif int(row.kernelId) < 0:
                    raise ValueError("Only parameterized OPUS rows may omit kernelId")
                if validate_saved and (pipeline or payload):
                    raise ValueError("pipeline/config metadata is only valid for OPUS rows")
                pipelines.append("")
                configs.append("")
        df["pipeline"], df["config"] = pipelines, configs
        return df

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        df = super().get_tuned_gemm_list(tuned_gemm_file, columns)
        if df.empty:
            return df.reindex(columns=self.columns)
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
        self.opus_families = None
        self.opus_pipelines = None
        self.opus_configs = ()
        self.flydsl_kids = getattr(args, "flydsl_kids", None)
        self.flydsl_bmm_kernels = getattr(args, "flydsl_bmm_kernels", None)
        self.flydsl_bmm_search = getattr(args, "flydsl_bmm_search", "balanced")
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
        self.opus_pipelines = getattr(args, "opus_pipelines", None)
        if getattr(args, "opus_configs", None):
            self.opus_configs = load_opus_configs(args.opus_configs)
        requested_families = getattr(args, "opus_families", None)
        if requested_families is not None:
            self.opus_families = frozenset(
                value.strip() for value in requested_families.split(",") if value.strip()
            )
            unknown = self.opus_families.difference(A8W8_BPRESHUFFLE_FAMILIES)
            if unknown or not self.opus_families:
                self.parser.error(
                    f"Invalid OPUS pipeline families: {sorted(unknown)}; "
                    f"available: {sorted(A8W8_BPRESHUFFLE_FAMILIES)}"
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

    def _candidate_configs(self, gfx, m, n, k):
        configs = pipeline_configs(
            (m, n, k), getattr(self, "opus_pipelines", None), gfx=gfx,
            include_legacy=self.opus_kids is not None,
            families=getattr(self, "opus_families", None),
        )
        extras = tuple(
            config for config in getattr(self, "opus_configs", ())
            if gfx == "gfx950" and config_supports_shape(config, m, n, k)
            and (getattr(self, "opus_pipelines", None) is None or config.pipeline in self.opus_pipelines)
        )
        configs = tuple({(config.pipeline, config.to_json()): config
                         for config in (*configs, *extras)}.values())
        return (
            configs
            if self.opus_kids is None
            else tuple(config for config in configs if config.legacy_kid in self.opus_kids)
        )

    def _candidate_kids(self, gfx, m, n, k):
        """Bridge parameter enumeration to the existing private launcher ABI."""
        return [config.legacy_kid for config in self._candidate_configs(gfx, m, n, k)]

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
        prepare_flydsl=False,
        compare=compare_outputs,
    ):
        return (
            info,
            gen_data,
            (m, n, k, seed, True) if prepare_flydsl else (m, n, k, seed),
            func,
            (arg_keys, *extra_args),
            dict(run_kwargs),
            run_torch,
            (ref_keys,),
            {"with_bounds": True},
            None,
            1e-2,
            1e-2,
            compare,
            None,
            ("out_bmm",) if info[4] == "flydsl_bmm" else ("out",),
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
        gfx, cu_num, m, n, k = info_keys
        if not preshuffleB:
            return []
        tasks = []
        for config in self._candidate_configs(gfx, m, n, k):
            kid = config.legacy_kid
            kernel = kernel_instance_from_config(config)
            for split_k in bpreshuffle_candidate_split_k(kernel, m, n, k, cu_num):
                info = (info_keys, config, split_k, kernel.name, "opus", True)
                tasks.append(
                    self._make_task(
                        info,
                        m,
                        n,
                        k,
                        seed,
                        run_config_bench,
                        _BENCH_KEYS,
                        (config, split_k),
                        run_kwargs,
                        prepare_flydsl=getattr(self, "_prepare_flydsl", False),
                    )
                )
        return tasks

    def get_gemm_a8w8_blockscale_flydsl_bmm_tune_task(
        self, info_keys, seed, preshuffleB, run_kwargs
    ):
        gfx, cu_num, m, n, k = info_keys
        if gfx != "gfx950" or not preshuffleB:
            return []
        runtime = flydsl_modules("flydsl_bmm")
        names = flydsl_bmm_candidate_names(
            runtime, m, n, k, cu_num, getattr(self, "flydsl_bmm_kernels", None),
            getattr(self, "flydsl_bmm_search", "balanced"),
        )
        return [self._make_task(
            (info_keys, "bmm", runtime.parse_bmm_kernel_name(name)["splits"], name, "flydsl_bmm", True),
            m, n, k, seed, run_flydsl_bmm_bench, _FLYDSL_BMM_KEYS, (name,), run_kwargs,
            prepare_flydsl=True, compare=compare_flydsl_outputs,
        ) for name in names]

    def get_gemm_a8w8_blockscale_flydsl_tune_task(
        self, info_keys, seed, preshuffleB, run_kwargs
    ):
        gfx, _, m, n, k = info_keys
        if gfx != "gfx950" or not preshuffleB:
            return []
        _, catalog = flydsl_modules("flydsl")
        kids = getattr(self, "flydsl_kids", None)
        if kids is not None and kids.difference(catalog.kernels_list):
            raise ValueError(f"Unknown FlyDSL numeric IDs: {sorted(kids.difference(catalog.kernels_list))}")
        # Reuse the common tuner's numeric catalog, shape/split filtering and
        # exact-name dispatch. Adapt only data/reference hooks to share native
        # scale values and operand storage with OPUS and BMM.
        original = super().get_gemm_a8w8_blockscale_flydsl_tune_task(
            info_keys, seed, preshuffleB, run_kwargs
        )
        if not original and catalog.candidates_for("fp8", "fp8", m, n, k):
            raise RuntimeError("The common tuner's FlyDSL catalog is unavailable; restart with a compatible FlyDSL environment")
        tasks = []
        for task in original:
            info = task[0]
            kid, kernel = info[1], catalog.kernels_list[info[1]]
            if kids is not None and kid not in kids:
                continue
            if (kernel.a_dtype, kernel.b_dtype, kernel.out_dtype) != ("fp8", "fp8", "bf16"):
                continue
            tasks.append(self._make_task(
                info, m, n, k, seed, task[3],
                ("x", "w", "x_scale_shuf", "w_scale_shuf", "out"), (kernel.name,), run_kwargs,
                prepare_flydsl=True, compare=compare_flydsl_outputs,
            ))
        return tasks

    def tune(self, untunedf, tunedf, args):
        self.opus_kids = getattr(args, "opus_kids", None)
        self.opus_pipelines = getattr(args, "opus_pipelines", None)
        self.flydsl_kids = getattr(args, "flydsl_kids", None)
        self.flydsl_bmm_kernels = getattr(args, "flydsl_bmm_kernels", None)
        self.flydsl_bmm_search = getattr(args, "flydsl_bmm_search", "balanced")
        backends = selected_backends(args.libtype)
        self._prepare_flydsl = bool(backends.intersection(("flydsl", "flydsl_bmm")))
        # Missing FlyDSL must fail before OPUS compilation or any GPU work.
        for backend in ("flydsl_bmm", "flydsl"):
            if backend in backends:
                flydsl_modules(backend)
        requested_configs = {}
        if "opus" in backends:
            for row in untunedf.itertuples(index=False):
                configs = self._candidate_configs(row.gfx, row.M, row.N, row.K)
                if args.libtype == "opus" and not configs:
                    raise ValueError(
                        f"No OPUS MXFP8 bpreshuffle candidates for "
                        f"{(row.M, row.N, row.K)}"
                    )
                requested_configs.update({(config.pipeline, config.to_json()): config for config in configs})
        if requested_configs:
            prepare_opus_configs(requested_configs.values())

        tasks, tasks_data = [], []
        run_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}
        gfx, cu_num = self.get_gfx(), self.get_cu_num()
        for row in untunedf.itertuples(index=False):
            info = (gfx, cu_num, row.M, row.N, row.K)
            start = len(tasks)
            if "ck" in backends:
                tasks.extend(self.get_gemm_a8w8_blockscale_tune_task(
                    info, args.splitK, 0, True, run_kwargs
                ))
            if "cktile" in backends:
                tasks.extend(self.get_gemm_a8w8_blockscale_cktile_tune_task(
                    info, args.splitK, 0, True, args.blockPerCu, run_kwargs
                ))
            if "asm" in backends:
                tasks.extend(self.get_gemm_a8w8_blockscale_asm_tune_task(
                    info, args.splitK, 0, True, run_kwargs
                ))
            if "opus" in backends:
                tasks.extend(self.get_gemm_a8w8_blockscale_opus_tune_task(
                    info, 0, True, run_kwargs
                ))
            if "flydsl_bmm" in backends:
                tasks.extend(self.get_gemm_a8w8_blockscale_flydsl_bmm_tune_task(
                    info, 0, True, run_kwargs
                ))
            if "flydsl" in backends:
                tasks.extend(self.get_gemm_a8w8_blockscale_flydsl_tune_task(
                    info, 0, True, run_kwargs
                ))
            if len(tasks) == start:
                raise ValueError(f"No {args.libtype} candidates support {(row.M, row.N, row.K)}")
            logger.info(
                f"MXFP8 {(row.M, row.N, row.K)}: {len(tasks) - start} candidates across "
                f"{', '.join(sorted(backends))}"
            )
            if len(tasks) != start:
                tasks_data.append((len(tasks) - start, ()))
        if not tasks:
            return []

        # In the unchanged mp_tuner, fast_mode=True refreshes the reference
        # when the generator changes. Every task supplies its own reference;
        # CK/CKTile/ASM/FlyDSL retain checkAllclose; OPUS uses bounds.
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
        # The generic profiler sorts the info tuple. Canonical configurations
        # deliberately have no ordering; sort their complete identities here.
        rets.sort(key=lambda result: (
            result[0][0], result[0][4],
            (result[0][1].pipeline, result[0][1].to_json())
            if isinstance(result[0][1], BpreshuffleConfig) else ("", str(result[0][1])),
            result[0][2],
        ))
        # Save the original timings and measured error ratios without alteration.
        # Suppress generic tuple sorting while retaining the requested profile.
        raw_args = argparse.Namespace(**vars(args))
        raw_args.profile_file = ""
        raw = super().post_process(rets, raw_args, topk=-1, fast_mode=True)
        if args.profile_file:
            profile = self.result_to_df(rets)
            if Path(args.profile_file).is_file():
                profile = pd.concat([pd.read_csv(args.profile_file), profile], ignore_index=True)
            profile.to_csv(args.profile_file, index=False, na_rep="Null")
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
            if isinstance(kernel_id, BpreshuffleConfig):
                return kernel_instance_from_config(kernel_id).name
            kernel = a8w8_mxscale_gemm_bpreshuffle_kernels_list.get(kernel_id)
            return None if kernel is None else kernel.name
        if libType == "flydsl":
            _, catalog = flydsl_modules("flydsl")
            kernel = catalog.kernels_list.get(kernel_id)
            return None if kernel is None else kernel.name
        if libType == "flydsl_bmm":
            # The bmm sentinel has no unique launch parameters; saved tasks
            # always supply their full kernelName.
            return None
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
        configs = [info[1] if isinstance(info[1], BpreshuffleConfig) else None
                   for info, _, _ in normalized]
        result = super().result_to_df(normalized)
        for index, config in enumerate(configs):
            if config is not None:
                result.at[index, "kernelId"] = config.legacy_kid
                result.at[index, "pipeline"] = config.pipeline
                result.at[index, "config"] = config.to_json()
        return self._config_columns(result)[self.columns]

    def run_config(self, args):
        from aiter.test_common import checkAllclose, run_perftest

        results = []
        rows = self._normalize_rows(self.untunedf, saved=True)
        unsupported = set(rows.libtype).difference(_SUPPORTED_LIBTYPES)
        if unsupported:
            raise ValueError(
                f"Saved MXFP8 replay does not support libtype(s): {sorted(unsupported)}"
            )
        replay_configs = {}
        runtimes = {}
        for backend in ("flydsl_bmm", "flydsl"):
            if backend in set(rows.libtype):
                runtimes[backend] = flydsl_modules(backend)
        for row in rows.itertuples(index=False):
            if row.gfx != self.get_gfx() or row.cu_num != self.get_cu_num():
                raise ValueError(
                    "Saved row does not match the current GPU's gfx/cu_num"
                )
            if row.libtype == "opus":
                selected_config = validate_saved_config(row.pipeline, row.config, int(row.kernelId))
                instance = kernel_instance_from_config(selected_config)
                if not config_supports_shape(selected_config, row.M, row.N, row.K):
                    raise ValueError(f"Saved OPUS kid {row.kernelId} is incompatible with {row}")
                try:
                    if row.splitK < 0:
                        raise ValueError("saved splitK must be an exact count or historical zero")
                    bpreshuffle_launch_plan(
                        instance,
                        row.M, row.N, row.K, int(row.splitK), int(row.cu_num),
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Saved OPUS kid {row.kernelId} / splitK={row.splitK} is incompatible with {row}"
                    ) from exc
                replay_configs[(selected_config.pipeline, selected_config.to_json())] = selected_config
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
            if row.libtype == "flydsl_bmm":
                if pd.isna(saved_name) or not str(saved_name).strip():
                    raise ValueError("Saved FlyDSL BMM row requires kernelName")
                validate_flydsl_bmm(
                    runtimes["flydsl_bmm"], saved_name, row.M, row.N, row.K, row.splitK
                )
            elif row.libtype == "flydsl":
                _, catalog = runtimes["flydsl"]
                kernel = catalog.kernels_list.get(row.kernelId)
                if kernel is None or (kernel.a_dtype, kernel.b_dtype, kernel.out_dtype) != ("fp8", "fp8", "bf16"):
                    raise ValueError(f"Saved FlyDSL numeric ID {row.kernelId} is incompatible")
                if not catalog.fits_shape(kernel, row.M, row.N, row.K):
                    raise ValueError(f"Saved FlyDSL numeric ID {row.kernelId} does not support the shape")
                if saved_name != kernel.name:
                    raise ValueError("Saved FlyDSL kernelName does not match numeric ID")
                if row.splitK != kernel.split_k:
                    raise ValueError("Saved FlyDSL splitK does not match numeric ID/kernelName")
            elif row.libtype == "asm":
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
                expected_name = (instance.name if row.libtype == "opus" else
                                 self.getKernelName(row.kernelId, row.libtype, True))
                if saved_name != expected_name:
                    raise ValueError(
                        f"Saved {row.libtype} kernelName does not match kid {row.kernelId}"
                    )
        if replay_configs:
            prepare_opus_configs(replay_configs.values())
        prepare_native = bool(runtimes)
        for row in rows.itertuples(index=False):
            kid = row.kernelId
            native = row.libtype in _NATIVE_LIBTYPES
            if native:
                gen_data, ref_keys = generate_data, _REF_KEYS
            else:
                gen_data, ref_keys = generic_tune.generate_data, _CK_REF_KEYS
            data = (gen_data(row.M, row.N, row.K, 0, prepare_native, device="cuda")
                    if native and prepare_native else gen_data(row.M, row.N, row.K, 0, device="cuda"))
            ref_inputs = tuple(data[key] for key in ref_keys)
            ref = (
                run_torch(*ref_inputs, with_bounds=True)
                if native
                else generic_tune.run_torch(*ref_inputs)
            )
            data["out"].fill_(float("nan"))
            if row.libtype == "opus":
                selected_config = validate_saved_config(row.pipeline, row.config, int(row.kernelId))
                bench, bench_args = run_config_bench, (
                    *(data[key] for key in _BENCH_KEYS),
                    selected_config,
                    row.splitK,
                )
            elif row.libtype in ("flydsl", "flydsl_bmm"):
                bench = run_flydsl_bmm_bench if row.libtype == "flydsl_bmm" else run_flydsl_bench
                keys = _FLYDSL_BMM_KEYS if row.libtype == "flydsl_bmm" else _FLYDSL_KEYS
                bench_args = (*(data[key] for key in keys), row.kernelName)
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
            elif native:
                error = compare_flydsl_outputs(
                    ref, out, printLog=args.verbose, tol_err_ratio=args.errRatio,
                )
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
                        f"libtype={row.libtype},kid={kid},splitK={row.splitK}"
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
