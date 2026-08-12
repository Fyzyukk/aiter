#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Isolated Task1/Task2 OPUS interface performance comparison.

Run each endpoint in a fresh process with ``AITER_JIT_DIR`` pointing at the
matching preserved JIT module.  The benchmark keeps tensors, actual kids and
split-K fixed and reports four increasingly narrow timing layers:

* public/high-level Python adapter;
* ``compile_ops`` raw binding;
* direct pybind with pre-converted ``aiter_tensor_t`` handles;
* captured graph replay (device work only).

Task1 uses the B0 ABI (``opus_gemm_a16w16_tune`` and generic ``opus_gemm``).
Task2 uses the four canonical family entries.  No endpoint is rebuilt.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import pathlib
import statistics
import sys
import time
from collections.abc import Callable

_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import torch
from torch import Tensor

from aiter import dtypes
from aiter.jit.core import compile_ops
from aiter.ops.opus import gemm_op_a16w16 as a16
from aiter.ops.opus import gemm_op_a8w8 as a8
from aiter.utility.dtypes import torch_to_aiter_pybind


A16_KID = 200
A16_SPLIT_K = 2
A16_M = 64
A16_N = 64
A16_K = 2048

A8_M = 256
A8_N = 256
A8_K = 256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", choices=("task1", "task2"), required=True)
    parser.add_argument("--pass-id", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iters", type=int, default=100)
    return parser.parse_args()


# Historical B0 bindings.  They are lazy, so defining them while running the
# Task2 endpoint does not try to resolve removed symbols.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_tune",
    develop=True,
)
def _task1_a16_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    bias: Tensor | None,
    workspace: Tensor | None,
    kernelId: int,
    splitK: int,
) -> Tensor: ...


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm", develop=True)
def _task1_generic_raw(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Tensor | None = None,
    x_scale: Tensor | None = None,
    w_scale: Tensor | None = None,
    bias: Tensor | None = None,
) -> Tensor: ...


def _measure(
    call: Callable[[], object],
    warmup: int,
    rounds: int,
    iters: int,
    *,
    stream: torch.cuda.Stream | None = None,
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


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _to_aiter(tensor: Tensor | None):
    return None if tensor is None else torch_to_aiter_pybind(tensor)


def _set_module_stream(module, stream: torch.cuda.Stream | None = None) -> None:
    live_stream = torch.cuda.current_stream() if stream is None else stream
    module._set_current_hip_stream(live_stream.cuda_stream)


def _capture_direct(
    module,
    call: Callable[[], object],
) -> tuple[torch.cuda.CUDAGraph, torch.cuda.Stream]:
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _set_module_stream(module, stream)
        for _ in range(3):
            call()
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    stream.synchronize()
    return graph, stream


def _task1_explicit(
    raw: Callable[..., object],
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    workspace: Tensor,
) -> Tensor:
    """B0 valid-call path, without the legacy positional compatibility branch."""
    batch, M, K = XQ.shape
    N = Y.shape[2]
    arch, cu_num = a16._device_arch_and_cu(XQ.device)
    config = a16.select_launch_config(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=batch,
        cu_num=cu_num,
        has_bias=False,
        input_dtype=XQ.dtype,
        output_dtype=Y.dtype,
        explicit_kid=A16_KID,
        explicit_split_k=A16_SPLIT_K,
    )
    if config.actual_kid != A16_KID:
        raise RuntimeError(f"Task1 explicit kid resolved to {config.actual_kid}")
    return a16._launch_a16w16_with_torch_workspace(
        raw,
        XQ,
        WQ,
        Y,
        None,
        config,
        workspace=workspace,
    )


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.module = importlib.import_module("module_deepgemm_opus")
        self.rows: list[dict[str, object]] = []

    def emit_case(
        self,
        *,
        case: str,
        family: str,
        layer: str,
        actual_kid: int,
        split_k: int | None,
        output_dtype: str,
        shape: list[int],
        call: Callable[[], object],
        check: Callable[[], None],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        samples = _measure(
            call,
            self.args.warmup,
            self.args.rounds,
            self.args.iters,
            stream=stream,
        )
        check()
        row: dict[str, object] = {
            "endpoint": self.args.endpoint,
            "pass": self.args.pass_id,
            "case": case,
            "family": family,
            "layer": layer,
            "actual_kid": actual_kid,
            "split_k": split_k,
            "output_dtype": output_dtype,
            "shape": shape,
            "warmup": self.args.warmup,
            "rounds": self.args.rounds,
            "iters": self.args.iters,
            "timing": _stats(samples),
        }
        self.rows.append(row)
        print("PERF_CASE " + json.dumps(row, sort_keys=True), flush=True)

    def emit_unavailable(self, *, family: str, reason: str) -> None:
        row = {
            "endpoint": self.args.endpoint,
            "pass": self.args.pass_id,
            "arch": "gfx950",
            "family": family,
            "actual_kid": None,
            "split_k": None,
            "status": "not executable: no registered gfx950 kernel",
            "reason": reason,
        }
        print("PERF_UNAVAILABLE " + json.dumps(row, sort_keys=True), flush=True)

    def run_a16(self) -> None:
        torch.manual_seed(0xB7600200)
        XQ = torch.randn(
            (1, A16_M, A16_K), device="cuda", dtype=torch.bfloat16
        )
        WQ = torch.randn(
            (1, A16_N, A16_K), device="cuda", dtype=torch.bfloat16
        )
        golden = torch.bmm(XQ.float(), WQ.float().transpose(1, 2))

        raw = (
            _task1_a16_raw
            if self.args.endpoint == "task1"
            else a16._opus_gemm_a16w16_launch_ctypes_raw
        )
        if self.args.endpoint == "task1":
            # The frozen Task1 module predates the C ABI symbol. Inject its
            # pybind binding into the current production-backend slot so the
            # otherwise unchanged shape-driven adapter remains benchmarkable.
            a16._opus_gemm_a16w16_launch_ctypes_raw = raw

        for dtype_name, dtype, rtol, atol in (
            ("bf16", torch.bfloat16, 0.03, 0.5),
            ("fp32", torch.float32, 1e-3, 0.05),
        ):
            Y = torch.empty(
                (1, A16_M, A16_N), device="cuda", dtype=dtype
            )
            workspace = torch.empty(
                (A16_SPLIT_K, 1, A16_M, A16_N),
                device="cuda",
                dtype=torch.float32,
            )

            def check() -> None:
                torch.cuda.synchronize()
                torch.testing.assert_close(Y.float(), golden, rtol=rtol, atol=atol)

            def high_level() -> Tensor:
                return a16.gemm_a16w16_opus(
                    XQ,
                    WQ,
                    dtype=dtype,
                    kernelId=A16_KID,
                    splitK=A16_SPLIT_K,
                    out=Y,
                )

            if self.args.endpoint == "task1":

                def explicit() -> Tensor:
                    return _task1_explicit(raw, XQ, WQ, Y, workspace)

            else:

                def explicit() -> Tensor:
                    return a16.opus_gemm_a16w16_launch(
                        XQ,
                        WQ,
                        Y,
                        kid=A16_KID,
                        split_k=A16_SPLIT_K,
                        workspace=workspace,
                    )

            def raw_call() -> object:
                return raw(
                    XQ,
                    WQ,
                    Y,
                    None,
                    workspace,
                    A16_KID,
                    A16_SPLIT_K,
                )

            XQ_a = _to_aiter(XQ)
            WQ_a = _to_aiter(WQ)
            Y_a = _to_aiter(Y)
            workspace_a = _to_aiter(workspace)
            if self.args.endpoint == "task1":

                def direct() -> object:
                    return self.module.opus_gemm_a16w16_tune(
                        XQ_a,
                        WQ_a,
                        Y_a,
                        None,
                        workspace_a,
                        A16_KID,
                        A16_SPLIT_K,
                    )

            else:

                def direct() -> object:
                    return self.module.opus_gemm_a16w16_launch(
                        XQ_a,
                        WQ_a,
                        Y_a,
                        None,
                        workspace_a,
                        A16_KID,
                        A16_SPLIT_K,
                    )

            common = {
                "family": "a16w16",
                "actual_kid": A16_KID,
                "split_k": A16_SPLIT_K,
                "output_dtype": dtype_name,
                "shape": [1, A16_M, A16_N, A16_K],
                "check": check,
            }
            self.emit_case(
                case=f"a16_high_level_{dtype_name}",
                layer="high_level_python",
                call=high_level,
                **common,
            )
            self.emit_case(
                case=f"a16_explicit_{dtype_name}",
                layer="explicit_python",
                call=explicit,
                **common,
            )
            self.emit_case(
                case=f"a16_raw_{dtype_name}",
                layer="compile_ops_raw",
                call=raw_call,
                **common,
            )
            _set_module_stream(self.module)
            self.emit_case(
                case=f"a16_direct_{dtype_name}",
                layer="direct_pybind_cpp",
                call=direct,
                **common,
            )
            graph, graph_stream = _capture_direct(self.module, direct)
            self.emit_case(
                case=f"a16_graph_{dtype_name}",
                layer="graph_replay",
                call=graph.replay,
                stream=graph_stream,
                **common,
            )

    def _a8_inputs(self):
        XQ = (
            torch.arange(A8_M * A8_K, device="cuda", dtype=torch.int32)
            .remainder(5)
            .sub(2)
            .reshape(1, A8_M, A8_K)
            .to(dtypes.fp8)
        )
        WQ = (
            torch.arange(A8_N * A8_K, device="cuda", dtype=torch.int32)
            .remainder(7)
            .sub(3)
            .reshape(1, A8_N, A8_K)
            .to(dtypes.fp8)
        )
        x_scale = torch.ones(
            (1, A8_M, A8_K // 128), device="cuda", dtype=torch.float32
        )
        x_scale[:, 1::2].mul_(0.5)
        x_scale[:, :, 1].mul_(2.0)
        w_scale = torch.ones(
            (1, A8_N // 128, A8_K // 128),
            device="cuda",
            dtype=torch.float32,
        )
        w_scale[:, 1].mul_(2.0)
        w_scale[:, :, 1].mul_(0.25)
        return XQ, WQ, x_scale, w_scale

    def run_a8_noscale(self, XQ: Tensor, WQ: Tensor) -> None:
        Y = torch.empty((1, A8_M, A8_N), device="cuda", dtype=torch.float32)
        golden = XQ[0].float() @ WQ[0].float().T

        def check() -> None:
            torch.cuda.synchronize()
            torch.testing.assert_close(Y[0], golden, rtol=0, atol=0)

        if self.args.endpoint == "task1":

            def public() -> Tensor:
                _task1_generic_raw(XQ, WQ, Y, None, None, None, None)
                return Y

            raw_call = public
        else:

            def public() -> Tensor:
                return a8.opus_gemm_a8w8_launch(XQ, WQ, Y, kid=2)

            def raw_call() -> object:
                return a8._opus_gemm_a8w8_launch_raw(XQ, WQ, Y, 2)

        XQ_a, WQ_a, Y_a = map(_to_aiter, (XQ, WQ, Y))
        if self.args.endpoint == "task1":

            def direct() -> object:
                return self.module.opus_gemm(
                    XQ_a, WQ_a, Y_a, None, None, None, None
                )

        else:

            def direct() -> object:
                return self.module.opus_gemm_a8w8_launch(XQ_a, WQ_a, Y_a, 2)

        common = {
            "family": "a8w8",
            "actual_kid": 2,
            "split_k": None,
            "output_dtype": "fp32",
            "shape": [1, A8_M, A8_N, A8_K],
            "check": check,
        }
        self.emit_case(
            case="a8_noscale_public",
            layer="family_public_python",
            call=public,
            **common,
        )
        self.emit_case(
            case="a8_noscale_raw",
            layer="compile_ops_raw",
            call=raw_call,
            **common,
        )
        _set_module_stream(self.module)
        self.emit_case(
            case="a8_noscale_direct",
            layer="direct_pybind_cpp",
            call=direct,
            **common,
        )
        graph, graph_stream = _capture_direct(self.module, direct)
        self.emit_case(
            case="a8_noscale_graph",
            layer="graph_replay",
            call=graph.replay,
            stream=graph_stream,
            **common,
        )

    def run_a8_blockscale(
        self,
        XQ: Tensor,
        WQ: Tensor,
        x_scale: Tensor,
        w_scale: Tensor,
    ) -> None:
        Y = torch.empty((1, A8_M, A8_N), device="cuda", dtype=torch.float32)
        golden = torch.zeros(
            (A8_M, A8_N), device="cuda", dtype=torch.float32
        )
        for block_k in range(A8_K // 128):
            partial = XQ[0, :, block_k * 128 : (block_k + 1) * 128].float() @ WQ[
                0, :, block_k * 128 : (block_k + 1) * 128
            ].float().T
            golden.add_(
                partial
                * x_scale[0, :, block_k].unsqueeze(1)
                * w_scale[0, :, block_k]
                .repeat_interleave(128)
                .unsqueeze(0)
            )

        def check() -> None:
            torch.cuda.synchronize()
            torch.testing.assert_close(Y[0], golden, rtol=0, atol=0)

        if self.args.endpoint == "task1":

            def public() -> Tensor:
                _task1_generic_raw(
                    XQ, WQ, Y, None, x_scale, w_scale, None
                )
                return Y

            raw_call = public
        else:

            def public() -> Tensor:
                return a8.opus_gemm_a8w8_blockscale_launch(
                    XQ, WQ, Y, x_scale, w_scale, kid=1
                )

            def raw_call() -> object:
                return a8._opus_gemm_a8w8_blockscale_launch_raw(
                    XQ, WQ, Y, x_scale, w_scale, 1
                )

        XQ_a, WQ_a, Y_a, x_scale_a, w_scale_a = map(
            _to_aiter, (XQ, WQ, Y, x_scale, w_scale)
        )
        if self.args.endpoint == "task1":

            def direct() -> object:
                return self.module.opus_gemm(
                    XQ_a,
                    WQ_a,
                    Y_a,
                    None,
                    x_scale_a,
                    w_scale_a,
                    None,
                )

        else:

            def direct() -> object:
                return self.module.opus_gemm_a8w8_blockscale_launch(
                    XQ_a, WQ_a, Y_a, x_scale_a, w_scale_a, 1
                )

        common = {
            "family": "a8w8_blockscale",
            "actual_kid": 1,
            "split_k": None,
            "output_dtype": "fp32",
            "shape": [1, A8_M, A8_N, A8_K],
            "check": check,
        }
        self.emit_case(
            case="a8_blockscale_public",
            layer="family_public_python",
            call=public,
            **common,
        )
        self.emit_case(
            case="a8_blockscale_raw",
            layer="compile_ops_raw",
            call=raw_call,
            **common,
        )
        _set_module_stream(self.module)
        self.emit_case(
            case="a8_blockscale_direct",
            layer="direct_pybind_cpp",
            call=direct,
            **common,
        )
        graph, graph_stream = _capture_direct(self.module, direct)
        self.emit_case(
            case="a8_blockscale_graph",
            layer="graph_replay",
            call=graph.replay,
            stream=graph_stream,
            **common,
        )

    def run(self) -> None:
        props = torch.cuda.get_device_properties(0)
        arch = str(getattr(props, "gcnArchName", "")).split(":", 1)[0].lower()
        if arch != "gfx950":
            raise RuntimeError(f"requires gfx950, got {arch!r}")

        module_path = pathlib.Path(self.module.__file__).resolve()
        exports = sorted(name for name in dir(self.module) if "opus" in name)
        if self.args.endpoint == "task1":
            required = {"opus_gemm", "opus_gemm_a16w16_tune"}
        else:
            required = {
                "opus_gemm_a16w16_launch",
                "opus_gemm_a8w8_launch",
                "opus_gemm_a8w8_blockscale_launch",
                "opus_gemm_a8w8_blockscale_bpreshuffle_launch",
            }
        missing = required - set(exports)
        if missing:
            raise RuntimeError(
                f"{self.args.endpoint} module is missing required exports {missing}"
            )

        start = {
            "endpoint": self.args.endpoint,
            "pass": self.args.pass_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": torch.cuda.get_device_name(0),
            "arch": str(getattr(props, "gcnArchName", None)),
            "module": str(module_path),
            "module_sha256": _sha256(module_path),
            "exports": exports,
            "warmup": self.args.warmup,
            "rounds": self.args.rounds,
            "iters": self.args.iters,
        }
        print("PERF_START " + json.dumps(start, sort_keys=True), flush=True)

        with torch.inference_mode():
            self.run_a16()
            XQ, WQ, x_scale, w_scale = self._a8_inputs()
            self.run_a8_noscale(XQ, WQ)
            self.run_a8_blockscale(XQ, WQ, x_scale, w_scale)
            if self.args.endpoint == "task1":
                reason = (
                    "B0 bpreshuffle tune was gfx942-only (kid 11000); gfx950 "
                    "had no kernel"
                )
            else:
                reason = "Task2 gfx950 bpreshuffle typed tables are empty"
            self.emit_unavailable(
                family="a8w8_blockscale_bpreshuffle", reason=reason
            )

        summary = {
            "endpoint": self.args.endpoint,
            "pass": self.args.pass_id,
            "cases": len(self.rows),
            "all_correct": True,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        print("PERF_COMPLETE " + json.dumps(summary, sort_keys=True), flush=True)


def main() -> None:
    args = _parse_args()
    if "AITER_JIT_DIR" not in os.environ:
        raise RuntimeError("AITER_JIT_DIR must point at a preserved endpoint module")
    Runner(args).run()


if __name__ == "__main__":
    main()
