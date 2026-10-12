#!/usr/bin/env python3
"""Precompile frozen BMM controls through the official CPU-only AOT path."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def no_gpu_context(torch):
    assert not torch.cuda.is_initialized()
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = fd.resolve()
        except FileNotFoundError:
            continue
        assert str(target) != "/dev/kfd", "CPU prebuild acquired a KFD context"


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    os.environ.update(plan["env"])
    os.environ.update(HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
                      COMPILE_ONLY="1", FLYDSL_GPU_ARCH="gfx950")
    sys.path.insert(0, plan["env"]["PYTHONPATH"])
    sys.path.insert(0, str(ROOT))
    # Serialize with the OPUS cache build and let the existing watcher defer
    # launch until this compilation phase also releases the same lock.
    lock = (HERE / "cache_build.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    import torch
    import flydsl
    from aiter.ops.flydsl.batched_gemm_a8w8_gfx950 import compile_bmm_a8w8_mxfp8_gfx950
    assert flydsl.__version__ == "0.3.4.1"
    assert "opus_flydsl_comparison_20261008/python_packages" in flydsl.__file__
    no_gpu_context(torch)
    for path, expected in {**plan["source_sha256"], **plan["input_sha256"]}.items():
        assert sha(ROOT / path) == expected, path
    wanted = {(1, 768, 7168), (16, 2048, 7168), (48, 7168, 16384),
              (112, 768, 7168), (672, 7168, 384), (1088, 7168, 768),
              (1024, 7168, 16384), (16384, 65536, 1536), (256, 7168, 7168)}
    cases = sorted(plan["cases"], key=lambda c: (c["M"], c["N"], c["K"]) not in wanted)
    jobs = {(c["flydsl_name"], c["N"], c["K"], c["M"] > 1): None for c in cases}
    destination = HERE / "flydsl_cache_build.json"
    assert not destination.exists(), "Preserve prior compilation receipt"
    rows = []
    try:
        for (name, n, k, transposed) in jobs:
            started = time.monotonic()
            row = {"name": name, "N": n, "K": k, "x_scale_transposed": transposed}
            try:
                compile_bmm_a8w8_mxfp8_gfx950(name, 1, n, k, 128, transposed)
                no_gpu_context(torch)
                row["returncode"] = 0
            except Exception as error:
                row.update(returncode=1, error=repr(error))
            row["seconds"] = time.monotonic()-started
            rows.append(row)
            save(destination, {"completed": len(rows), "total": len(jobs), "rows": rows,
                               "kernel_launches": 0, "gpu_contexts": 0,
                               "plan_sha256": sha(HERE / "plan.json"),
                               "script_sha256": sha(__file__)})
            print(json.dumps(row), flush=True)
        files = {str(p.relative_to(HERE)): sha(p)
                 for p in (HERE / "flydsl_cache").rglob("*") if p.is_file()}
        save(HERE / "flydsl_cache_manifest.json", {"files": files,
             "build_receipt_sha256": sha(destination), "kernel_launches": 0})
        return int(any(row["returncode"] for row in rows))
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
