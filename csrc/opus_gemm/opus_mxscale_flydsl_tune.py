# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL MXPSH catalog and baseline-derived B=1 BMM tuning tasks."""

import csv
import functools
import math
from pathlib import Path

DEFAULT_BASELINE = str(
    Path(__file__).resolve().parents[2]
    / "aiter/configs/model_configs/dsv4_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"
)
MXPSH_KEYS = ("x", "weight_shuffle", "x_scale_shuf", "w_scale_shuf", "out")
BMM_KEYS = ("x", "weight_shuffle", "x_scale_bmm", "w_scale_bmm", "out")
OPUS_REF_KEYS = ("reference_bounds",)
FLYDSL_REF_KEYS = ("reference_bf16",)


@functools.lru_cache(maxsize=8)
def load_baseline(path):
    """Keep both operand contracts at a shape, including string BMM IDs."""
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"gfx", "M", "N", "K", "libtype", "kernelId", "splitK", "kernelName"}
    if rows and required.difference(rows[0]):
        raise ValueError(f"FlyDSL baseline missing columns: {sorted(required.difference(rows[0]))}")
    return tuple(
        row for row in rows
        if row["gfx"] == "gfx950" and row["libtype"] == "flydsl"
        and row.get("w_scale_block", "128x128") == "128x128"
    )


def validate_candidate(kernel_id, name, split_k, m, n, k):
    """Validate the launch contract, allowing historical names outside the catalog."""
    if kernel_id == "bmm":
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

    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
        instance_valid,
        kernelInstance,
        parse_kernel_name,
    )

    cfg = parse_kernel_name(name)
    if cfg is None or (cfg["a_dtype"], cfg["b_dtype"], cfg["out_dtype"]) != ("fp8", "fp8", "bf16"):
        raise ValueError(f"Invalid FlyDSL a8w8 BF16 MXPSH name: {name!r}")
    if cfg["split_k"] != split_k or not instance_valid(kernelInstance(**cfg)):
        raise ValueError(f"Invalid FlyDSL MXPSH config/splitK: {name!r}/{split_k}")
    if min(m, n, k) <= 0 or n % 128 or k % 128 or n % cfg["tile_n"] or k % cfg["tile_k"]:
        raise ValueError(f"FlyDSL MXPSH {name!r} does not support {(m, n, k)}")
    if cfg["tile_m"] == 16 and m > 16:
        raise ValueError("FlyDSL tile_m=16 requires M<=16")
    if split_k > 1 and (
        k % split_k or (k // split_k) % cfg["tile_k"]
        or (k // split_k) % 256 or m * n * split_k * 4 >= (1 << 32)
    ):
        raise ValueError("Illegal FlyDSL MXPSH split-K partition/workspace")
    return cfg


def bmm_candidates(m, n, k, baseline_path=DEFAULT_BASELINE):
    """All legal baseline BMM names plus the latest per-shape heuristic.

    Upstream supplies no finite BMM tuning catalog. The baseline name union is
    an explicit, reproducible search space; it retains every saved baseline
    candidate and includes the current fallback for shapes without a saved row.
    """
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import (
        parse_bmm_kernel_name,
        pick_bmm_kernel_name,
    )

    rows = load_baseline(str(baseline_path))
    names = {row["kernelName"] for row in rows if row["kernelId"] == "bmm"}
    names.add(pick_bmm_kernel_name(1, m, n, k, 128, 128, 128, m > 1))
    result = []
    for name in sorted(names):
        cfg = parse_bmm_kernel_name(name)
        if cfg is None:
            raise ValueError(f"Unrecognized FlyDSL baseline BMM name: {name!r}")
        try:
            validate_candidate("bmm", name, cfg["splits"], m, n, k)
        except ValueError:
            continue
        result.append(("bmm", cfg["splits"], name))
    # A malformed or newly incompatible own-shape row must never disappear.
    for row in rows:
        if row["kernelId"] == "bmm" and (int(row["M"]), int(row["N"]), int(row["K"])) == (m, n, k):
            validate_candidate("bmm", row["kernelName"], int(row["splitK"]), m, n, k)
    return result


def generate_data(m, n, k, seed, device="cuda"):
    """Shared native OPUS E8M0 inputs, caller-prepared layouts and one oracle.

    A scales are logical [M,K/128] with column-major storage; B scales are
    row-major [N/128,K/128]. BMM views those same bytes. MXPSH needs a layout
    shuffle of the logical scales, prepared here before any GEMM is timed.
    No scale dtype conversion or FP32-to-E8M0 quantization occurs.
    """
    from aiter.ops.shuffle import (
        shuffle_scale_blockscale_a,
        shuffle_scale_blockscale_b,
    )
    from csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune import (
        generate_data as generate_native_data,
    )
    from csrc.opus_gemm.opus_gemm_mxscale_bpreshuffle_tune import run_torch

    if n % 128:
        raise ValueError("Shared OPUS/FlyDSL blockscale inputs require N divisible by 128")
    data = generate_native_data(m, n, k, seed, device=device)
    data["weight_shuffle"] = data["w"]
    data["x_scale_bmm"] = data["x_scale"].T.view(m, k // 128)
    data["w_scale_bmm"] = data["w_scale"]
    data["x_scale_shuf"] = shuffle_scale_blockscale_a(data["x_scale"], k)
    data["w_scale_shuf"] = shuffle_scale_blockscale_b(data["w_scale"], n, k)
    data["reference_bounds"] = run_torch(
        data["x"], data["w_reference"], data["x_scale"], data["w_scale"],
        with_bounds=True,
    )
    data["reference_bf16"] = data["reference_bounds"][0].to(data["out"].dtype)
    return data


def reference_from_data(reference):
    """Return the caller-prepared oracle; switching backend adds no matmul."""
    return reference


def run_bmm(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import run_bmm_a8w8_mxfp8_gfx950

    m, k = x.shape
    n = out.shape[1]
    run_bmm_a8w8_mxfp8_gfx950(
        x.view(m, 1, k), weight_shuffle.view(1, n, k),
        x_scale.view(m, 1, k // 128), w_scale.view(1, n // 128, k // 128),
        out.view(m, 1, n), kernel_name=kernel_name, x_scale_transposed=True,
    )
    return out


def get_baseline_tasks(info_keys, seed, run_kwargs, baseline_path=DEFAULT_BASELINE):
    """Replay only the exact saved FlyDSL identities of this shape as candidates."""
    from csrc.ck_gemm_a8w8_blockscale import gemm_a8w8_blockscale_tune as generic

    gfx, _, m, n, k = info_keys
    if gfx != "gfx950":
        return []
    tasks = []
    for row in load_baseline(str(baseline_path)):
        if (int(row["M"]), int(row["N"]), int(row["K"])) != (m, n, k):
            continue
        kid = "bmm" if row["kernelId"] == "bmm" else int(row["kernelId"])
        sk, name = int(row["splitK"]), row["kernelName"]
        validate_candidate(kid, name, sk, m, n, k)
        runner, keys = (
            (run_bmm, BMM_KEYS) if kid == "bmm"
            else (generic.run_gemm_a8w8_blockscale_flydsl, MXPSH_KEYS)
        )
        tasks.append((
            (info_keys, kid, sk, name, "flydsl", True), generate_data, (m, n, k, seed),
            runner, (keys, name), dict(run_kwargs), reference_from_data,
            (FLYDSL_REF_KEYS,), {}, None, 1e-2, 0.01, None, None, ("out",),
        ))
    if not tasks:
        raise ValueError(f"No exact FlyDSL baseline identities for {(m, n, k)}")
    return tasks


def get_tune_tasks(tuner, info_keys, seed, run_kwargs, baseline_path=DEFAULT_BASELINE):
    """Reuse upstream MXPSH tasks and accuracy/reference contract for both paths."""
    from csrc.ck_gemm_a8w8_blockscale import gemm_a8w8_blockscale_tune as generic

    gfx, _, m, n, k = info_keys
    if gfx != "gfx950":
        return []
    tasks = tuner.get_gemm_a8w8_blockscale_flydsl_tune_task(info_keys, seed, True, run_kwargs)
    if not tasks:
        raise RuntimeError("Latest FlyDSL MXPSH catalog is unavailable or has no legal candidates")
    # Keep the upstream BF16 oracle and tolerances, using the shared native
    # reference plane already computed for OPUS's accumulation-bounds check.
    tasks = [
        (task[0], generate_data, *task[2:6], reference_from_data,
         (FLYDSL_REF_KEYS,), {}, *task[9:])
        for task in tasks
    ]
    present = {task[0][3] for task in tasks}

    def make_task(kid, sk, name, runner, keys):
        return (
            (info_keys, kid, sk, name, "flydsl", True), generate_data, (m, n, k, seed),
            runner, (keys, name), dict(run_kwargs), reference_from_data,
            (FLYDSL_REF_KEYS,), {}, None,
            1e-2, 0.01, None, None, ("out",),
        )

    for row in load_baseline(str(baseline_path)):
        if row["kernelId"] == "bmm" or (int(row["M"]), int(row["N"]), int(row["K"])) != (m, n, k):
            continue
        kid, sk, name = int(row["kernelId"]), int(row["splitK"]), row["kernelName"]
        validate_candidate(kid, name, sk, m, n, k)
        if name not in present:
            tasks.append(make_task(kid, sk, name, generic.run_gemm_a8w8_blockscale_flydsl, MXPSH_KEYS))
            present.add(name)
    for kid, sk, name in bmm_candidates(m, n, k, baseline_path):
        tasks.append(make_task(kid, sk, name, run_bmm, BMM_KEYS))
    return tasks


def candidate_counts(m, n, k, baseline_path=DEFAULT_BASELINE):
    """Report the declared spaces without preparing inputs or launching kernels."""
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
        candidates_for,
    )

    names = {ki.name for _, ki in candidates_for("fp8", "fp8", m, n, k)}
    for row in load_baseline(str(baseline_path)):
        if row["kernelId"] != "bmm" and (int(row["M"]), int(row["N"]), int(row["K"])) == (m, n, k):
            validate_candidate(int(row["kernelId"]), row["kernelName"], int(row["splitK"]), m, n, k)
            names.add(row["kernelName"])
    bmm_count = len(bmm_candidates(m, n, k, baseline_path))
    return {"mxpsh_candidates": len(names), "bmm_candidates": bmm_count, "flydsl_total": len(names) + bmm_count}
