#!/usr/bin/env python3
"""Compile a complete plan once before any GPU worker starts timing."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    kids = {candidate["kid"] for item in plan["shapes"] for candidate in item["candidates"]}
    torch.cuda.set_device(args.gpu)
    root = Path(__file__).resolve().parents[3]
    path = root / "csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py"
    spec = importlib.util.spec_from_file_location("_stage5_precompile", path)
    tuner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tuner)
    rebuilt = tuner.ensure_opus_kids_compiled(kids, cu_num=256)
    args.receipt.write_text(json.dumps({"candidate_kids": sorted(kids), "rebuilt": rebuilt,
                                      "gpu_benchmark_started": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
