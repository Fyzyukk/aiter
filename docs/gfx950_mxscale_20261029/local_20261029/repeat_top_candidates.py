#!/usr/bin/env python3
"""Repeat the CPU-generated m-repeat shortlist on one GPU with balanced order.

Default timing matches the tuner's profiler and automatic input rotation.
--rotate-args 1 intentionally switches to hot-cache timing and disables CSV wins.
Run only after the relevant tuning batch has finished; this script uses a GPU.
"""

import argparse
import importlib
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LOCAL = Path(__file__).resolve().parent


def summarize(records, args):
    groups = defaultdict(list)
    for record in records:
        groups[(tuple(record["shape"][field] for field in ("gfx", "cu_num", "M", "N", "K", "w_scale_block")),
                record["kid"])].append(record)
    summaries = []
    for (_, kid), rows in groups.items():
        valid = [row["us"] for row in rows if row["correct"] and row["us"] is not None and row["us"] > 0]
        median = statistics.median(valid) if valid else None
        baseline = rows[0]["csv_baseline_us"]
        all_correct = len(valid) == args.rounds
        summaries.append({"shape": rows[0]["shape"], "kid": kid, "kernelName": rows[0]["kernelName"],
                          "bounded": rows[0]["bounded"], "rounds": len(rows), "all_rounds_correct": all_correct,
                          "median_us": median, "min_us": min(valid) if valid else None,
                          "max_us": max(valid) if valid else None,
                          "pstdev_us": statistics.pstdev(valid) if len(valid) > 1 else 0 if valid else None,
                          "max_errRatio": max(row["errRatio"] for row in rows),
                          "csv_baseline_us": baseline,
                          "wins_csv": bool(all_correct and baseline is not None and median < baseline)
                          if args.rotate_args == 0 else None,
                          "csv_speedup": baseline / median if all_correct and baseline is not None and args.rotate_args == 0 else None,
                          "exact_opposite_kids": rows[0]["exact_opposite_kids"]})
    by_shape = defaultdict(dict)
    for row in summaries:
        by_shape[tuple(row["shape"].values())][row["kid"]] = row
    paired = []
    for candidates in by_shape.values():
        for row in candidates.values():
            if row["bounded"]:
                continue
            for opposite in row["exact_opposite_kids"]:
                right = candidates.get(opposite)
                if right is None:
                    continue
                left_rounds = {r["round"]: r for r in groups[(tuple(row["shape"][f] for f in ("gfx", "cu_num", "M", "N", "K", "w_scale_block")), row["kid"])]}
                right_rounds = {r["round"]: r for r in groups[(tuple(right["shape"][f] for f in ("gfx", "cu_num", "M", "N", "K", "w_scale_block")), right["kid"])]}
                ratios = [right_rounds[r]["us"] / left_rounds[r]["us"] for r in left_rounds.keys() & right_rounds.keys()
                          if left_rounds[r]["correct"] and right_rounds[r]["correct"]]
                paired.append({"shape": row["shape"], "false_kid": row["kid"], "true_kid": opposite,
                               "paired_correct_rounds": len(ratios),
                               "median_true_over_false": statistics.median(ratios) if ratios else None,
                               "median_false_advantage_pct": (statistics.median(ratios) - 1) * 100 if ratios else None})
    return {"timing_mode": "tuner profiler with automatic input rotation" if args.rotate_args == 0 else f"profiler with fixed rotation={args.rotate_args}; CSV comparison disabled",
            "rounds_requested": args.rounds, "iters": args.iters, "warmup": args.warmup,
            "gpu_logical_id": args.gpu, "candidates": summaries, "exact_pairs": paired}


def main(default_plan=None, default_output=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=default_plan or LOCAL / "step1_paired/repeat_plan.json")
    parser.add_argument("--output", type=Path, default=default_output or LOCAL / "step1_repeat.jsonl")
    parser.add_argument("--gpu", type=int, default=0, help="Logical index within visible GPU list")
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iters", type=int, default=101)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rotate-args", type=int, default=0)
    parser.add_argument("--max-shapes", type=int, default=0, help="0 means all plan shapes")
    parser.add_argument("--shape", help="Only M,N,K, such as 8192,768,7168")
    parser.add_argument("--dry-run", action="store_true", help="Validate plan and show counts without importing torch")
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or args.iters < 2 or args.warmup < 0 or args.rotate_args < 0 or args.max_shapes < 0:
        parser.error("Use positive even rounds, iters>=2, warmup>=0, rotation>=0, max-shapes>=0")
    plan = json.loads(args.plan.read_text())
    shapes = plan["shapes"]
    if args.shape:
        chosen = tuple(int(value) for value in args.shape.split(","))
        if len(chosen) != 3:
            parser.error("--shape must be M,N,K")
        shapes = [item for item in shapes if tuple(item["shape"][key] for key in ("M", "N", "K")) == chosen]
    if args.max_shapes:
        shapes = shapes[:args.max_shapes]
    if not shapes:
        parser.error("No selected shapes in plan")
    sys.path.insert(0, str(LOCAL))
    metadata_module = "prepare_opus_repeat" if int(plan.get("version", 1)) >= 2 else "paired_mrepeat"
    registry_metadata = importlib.import_module(metadata_module).registry_metadata
    metadata, pairs = registry_metadata()
    for item in shapes:
        if item["shape"]["gfx"] != "gfx950" or item["shape"]["w_scale_block"] != "128x128":
            parser.error("Plan must contain gfx950/128x128 shapes")
        for candidate in item["candidates"]:
            kid = int(candidate["kid"])
            current_traits = metadata[kid]["traits"] if kid in metadata else None
            if isinstance(current_traits, tuple):
                current_traits = list(current_traits)
            if (kid not in metadata or candidate["kernelName"] != metadata[kid]["kernelName"]
                    or candidate["traits"] != current_traits
                    or candidate["geometry"] != list(metadata[kid]["geometry"])):
                parser.error(f"Candidate kid {kid} identity differs from current registry")
    if args.dry_run:
        print(json.dumps({"shapes": len(shapes), "candidates": sum(len(item["candidates"]) for item in shapes),
                          "rounds": args.rounds, "torch_imported": "torch" in sys.modules,
                          "timing_matches_tuner": args.rotate_args == 0}, indent=2))
        return
    if args.output.exists():
        parser.error(f"Output exists; select a fresh path to preserve earlier rounds: {args.output}")
    os.environ["AITER_REBUILD"] = "0"
    sys.path.insert(0, str(ROOT))
    import torch
    from aiter.test_common import checkAllclose, run_perftest
    path = ROOT / "csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py"
    spec = importlib.util.spec_from_file_location("_repeat_tuner", path)
    tuner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner)
    torch.cuda.set_device(args.gpu)
    props = torch.cuda.get_device_properties(args.gpu)
    import ctypes
    hip = ctypes.CDLL("/opt/rocm/lib/libamdhip64.so")
    pci_buffer = ctypes.create_string_buffer(32)
    pci_fn = hip.hipDeviceGetPCIBusId
    pci_fn.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    pci_fn.restype = ctypes.c_int
    pci_status = pci_fn(pci_buffer, 32, args.gpu)
    if pci_status != 0:
        parser.error(f"Could not verify GPU PCI bus: HIP status {pci_status}")
    gpu_pci_bus = pci_buffer.value.decode().lower()
    if plan.get("gpu_pci_bus") and gpu_pci_bus != plan["gpu_pci_bus"].lower():
        parser.error(f"GPU PCI bus differs from plan: {gpu_pci_bus} vs {plan['gpu_pci_bus']}")
    if not str(props.gcnArchName).startswith("gfx950"):
        parser.error(f"Expected gfx950 device; got {props.gcnArchName}")
    if any(int(item["shape"]["cu_num"]) != props.multi_processor_count for item in shapes):
        parser.error("Selected GPU CU count differs from repeat plan")
    kids = {int(candidate["kid"]) for item in shapes for candidate in item["candidates"]}
    tuner.ensure_opus_kids_compiled(kids, cu_num=props.multi_processor_count)
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream, torch.inference_mode():
        for item in shapes:
            shape = item["shape"]
            data = tuner.generate_data(shape["M"], shape["N"], shape["K"], 0,
                                       "128x128", device=f"cuda:{args.gpu}")
            reference = tuner.run_torch(data["x_deq"], data["w_deq"])
            call_args = tuple(data[key] for key in ("x", "weight_shuffle", "x_scale", "w_scale", "out"))
            candidates = item["candidates"]
            def correctness(kid):
                data["out"].fill_(float("nan"))
                torch.cuda.synchronize()
                tuner.run_gemm_opus(*call_args, kid)
                torch.cuda.synchronize()
                if not bool(torch.isfinite(data["out"]).all()):
                    return 1.0
                return float(checkAllclose(reference, data["out"], rtol=1e-2, atol=0.01,
                                           tol_err_ratio=plan["correctness_limit"], printLog=False,
                                           catastrophic_check=True))
            initial_errors = {candidate["kid"]: correctness(candidate["kid"]) for candidate in candidates}
            for round_index in range(args.rounds):
                offset = (round_index // 2) % len(candidates)
                order = candidates[offset:] + candidates[:offset]
                if round_index % 2:
                    order = list(reversed(order))
                for position, candidate in enumerate(order):
                    kid = candidate["kid"]
                    error = initial_errors[kid]
                    latency = None
                    failure = None
                    try:
                        error = max(error, correctness(kid))
                        if error <= plan["correctness_limit"]:
                            _, latency = run_perftest(tuner.run_gemm_opus, *call_args, kid,
                                                      num_iters=args.iters, num_warmup=args.warmup,
                                                      num_rotate_args=args.rotate_args)
                            torch.cuda.synchronize()
                            error = max(error, correctness(kid))
                            latency = float(latency)
                            if not math.isfinite(latency):
                                latency = None
                    except Exception as exc:
                        error, failure = 1.0, f"{type(exc).__name__}: {exc}"
                    record = {"completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                              "shape": shape, "round": round_index, "position": position,
                              "kid": kid, "kernelName": candidate["kernelName"], "bounded": candidate["bounded"],
                              "kind": candidate.get("kind", "JIT"), "roles": candidate.get("roles", []),
                              "us": latency, "errRatio": error,
                              "correct": bool(latency is not None and latency > 0 and error <= plan["correctness_limit"]),
                              "csv_baseline_us": item["csv_baseline_us"], "exact_opposite_kids": pairs[kid],
                              "rotate_args": args.rotate_args, "gpu_logical_id": args.gpu,
                              "gpu_physical_id": plan.get("gpu_physical_id", args.gpu), "failure": failure}
                    record["gpu_pci_bus"] = gpu_pci_bus
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    records.append(record)
                    print(json.dumps({"M": shape["M"], "N": shape["N"], "K": shape["K"],
                                      "round": round_index, "kid": kid, "us": latency, "correct": record["correct"]}), flush=True)
            del reference, data, call_args
            torch.cuda.empty_cache()
    report = summarize(records, args)
    report_path = args.output.with_suffix(".summary.json")
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"measurements": len(records), "summary": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
