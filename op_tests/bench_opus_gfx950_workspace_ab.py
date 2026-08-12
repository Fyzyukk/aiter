#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Isolated gfx950 benchmark for baseline, pybind, ctypes, and public paths.

Run each endpoint in a fresh process with the matching PYTHONPATH and JIT
directory. The script intentionally benchmarks the raw binding so the old
internal-workspace baseline, caller-owned pybind path, C ABI/ctypes raw, and
the adopted public canonical path can be compared with identical tensors and
launch geometry. ``current`` is retained as the historical pybind endpoint
label used by earlier logs.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import statistics
import time

import torch

import aiter
from aiter.ops.opus import gemm_op_a16w16 as gemm
from csrc.opus_gemm.opus_gemm_common import kernels_list


WORKSPACE_KIDS = tuple(range(200, 224)) + tuple(range(1200, 1224))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        choices=(
            "baseline",
            "current",
            "ctypes",
            "public-pybind",
            "public",
        ),
        required=True,
    )
    parser.add_argument("--pass-id", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


def _launch(endpoint, xq, wq, y, workspace, kid):
    if endpoint == "baseline":
        gemm._opus_gemm_a16w16_tune_raw(xq, wq, y, None, kid, 2)
    elif endpoint == "current":
        gemm._opus_gemm_a16w16_launch_raw(xq, wq, y, None, workspace, kid, 2)
    elif endpoint == "ctypes":
        gemm._opus_gemm_a16w16_launch_ctypes_raw(
            xq, wq, y, None, workspace, kid, 2
        )
    elif endpoint == "public-pybind":
        gemm._explicit_a16w16_launch(
            gemm._opus_gemm_a16w16_launch_raw,
            xq,
            wq,
            y,
            None,
            kid,
            2,
            workspace=workspace,
        )
    else:
        gemm.opus_gemm_a16w16_launch(
            xq,
            wq,
            y,
            kid=kid,
            split_k=2,
            workspace=workspace,
        )


def _measure(
    call, warmup: int, rounds: int, iters: int, stream=None
) -> list[float]:
    stream_context = (
        torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    )
    with stream_context:
        for _ in range(warmup):
            call()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(rounds):
            start.record()
            for _ in range(iters):
                call()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)) * 1000.0 / iters)
    return samples


def _stats(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "median_us": statistics.median(samples),
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "samples_us": samples,
    }


def main() -> None:
    args = _parse_args()
    props = torch.cuda.get_device_properties(0)
    arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()
    if arch != "gfx950":
        raise RuntimeError(f"requires gfx950, got {arch!r}")

    start_record = {
        "endpoint": args.endpoint,
        "pass": args.pass_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(pathlib.Path(aiter.__file__).resolve()),
        "device": torch.cuda.get_device_name(0),
        "arch": str(getattr(props, "gcnArchName", None)),
        "warmup": args.warmup,
        "rounds": args.rounds,
        "iters": args.iters,
        "kids": list(WORKSPACE_KIDS),
    }
    print("PERF_START " + json.dumps(start_record, sort_keys=True), flush=True)

    rows = []
    graph_stream = torch.cuda.Stream()
    if args.endpoint == "baseline":
        # The old endpoint owns a stream-keyed internal workspace.  Register
        # the exact stream eagerly; each case below is also prewarmed on this
        # stream before capture so any required growth happens outside capture.
        with torch.cuda.stream(graph_stream):
            gemm.opus_gemm_workspace_init()
        graph_stream.synchronize()

    with torch.inference_mode():
        for kid in WORKSPACE_KIDS:
            inst = kernels_list[kid]
            m, n, k = int(inst.B_M), int(inst.B_N), 32 * int(inst.B_K)
            torch.manual_seed(0x950000 + kid)
            xq = torch.randn((1, m, k), device="cuda", dtype=torch.bfloat16)
            wq = torch.randn((1, n, k), device="cuda", dtype=torch.bfloat16)
            golden = torch.bmm(xq.float(), wq.float().transpose(1, 2))
            workspace = (
                None
                if args.endpoint == "baseline"
                else torch.empty((2, 1, m, n), device="cuda", dtype=torch.float32)
            )

            for dtype_name, dtype, rtol, atol in (
                ("bf16", torch.bfloat16, 0.03, 0.5),
                ("fp32", torch.float32, 1e-3, 0.05),
            ):
                y = torch.empty((1, m, n), device="cuda", dtype=dtype)

                def eager_call():
                    _launch(args.endpoint, xq, wq, y, workspace, kid)

                eager_samples = _measure(
                    eager_call, args.warmup, args.rounds, args.iters
                )
                torch.testing.assert_close(y.float(), golden, rtol=rtol, atol=atol)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(graph_stream):
                    eager_call()
                graph_stream.synchronize()
                with torch.cuda.graph(graph, stream=graph_stream):
                    eager_call()
                graph_samples = _measure(
                    graph.replay,
                    args.warmup,
                    args.rounds,
                    args.iters,
                    stream=graph_stream,
                )
                torch.testing.assert_close(y.float(), golden, rtol=rtol, atol=atol)

                eager_stats = _stats(eager_samples)
                graph_stats = _stats(graph_samples)
                row = {
                    "endpoint": args.endpoint,
                    "pass": args.pass_id,
                    "kid": kid,
                    "dtype": dtype_name,
                    "shape": [1, m, n, k],
                    "warmup": args.warmup,
                    "rounds": args.rounds,
                    "iters": args.iters,
                    "eager": eager_stats,
                    "graph": graph_stats,
                    "eager_tflops": (2.0 * m * n * k)
                    / (float(eager_stats["median_us"]) * 1e6),
                    "graph_tflops": (2.0 * m * n * k)
                    / (float(graph_stats["median_us"]) * 1e6),
                }
                rows.append(row)
                print("PERF_CASE " + json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "endpoint": args.endpoint,
        "pass": args.pass_id,
        "cases": len(rows),
        "median_eager_us": statistics.median(
            float(row["eager"]["median_us"]) for row in rows
        ),
        "median_graph_us": statistics.median(
            float(row["graph"]["median_us"]) for row in rows
        ),
        "sum_eager_us": sum(float(row["eager"]["median_us"]) for row in rows),
        "sum_graph_us": sum(float(row["graph"]["median_us"]) for row in rows),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("PERF_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    print("PERF_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
