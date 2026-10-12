#!/usr/bin/env python3
"""Prebuild exact runtime modules without importing compiled .so or using GPU."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def main():
    plan = json.loads((HERE / "plan.json").read_text())
    os.environ.update(plan["env"])
    os.environ.update(HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
    lock = (HERE / "cache_build.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # AOT import avoids auto-loading public extensions. _build_module builds
    # Python extensions without get_module or ensure_config's library load.
    from aiter.ops.opus import bpreshuffle_runtime as runtime
    from csrc.opus_gemm.opus_gemm_bpreshuffle_config import construct_config
    core = runtime._core()
    wanted = {(1, 768, 7168), (16, 2048, 7168), (48, 7168, 16384),
              (112, 768, 7168), (672, 7168, 384), (1088, 7168, 768),
              (1024, 7168, 16384), (16384, 65536, 1536), (256, 7168, 7168)}
    ordered = {}
    for case in plan["cases"]:
        if (case["M"], case["N"], case["K"]) not in wanted:
            continue
        seen = set()
        ordered[case["baseline"]["config"]] = plan["configs"][case["baseline"]["config"]]
        for task in case["candidates"]:
            c = plan["configs"][task["config"]]["compile_params"]
            pipeline = plan["configs"][task["config"]]["pipeline"]
            key = (pipeline, c["schedule"], c["b_direct"], c["fixed_k"] > 0, c["wave_k"],
                   task["split_k"] if c["runtime_split_k"] else c["split_k"],
                   c["c_chunk_rows"] if pipeline == "large_output" else 0)
            if key in seen or c["runtime_split_k"] and task["split_k"] not in (1, 2, 3):
                continue
            seen.add(key)
            ordered[task["config"]] = plan["configs"][task["config"]]
    ordered.update(plan["configs"])
    rows = []
    try:
        for identity, row in ordered.items():
            started = time.monotonic()
            config = construct_config(row["pipeline"], config=row["compile_params"])
            args = runtime._build_arguments(config, core)
            assert args["is_python_module"], "build must not dynamically load extensions"
            # core AITER_JIT_DIR uses build as bd_dir; final module is beside it.
            artifact = Path(plan["env"]["AITER_JIT_DIR"]) / (args["md_name"] + ".so")
            record = {"identity": identity, "pipeline": config.pipeline, "module": args["md_name"]}
            try:
                if not artifact.exists():
                    runtime._build_module(core, args)
                assert artifact.is_file()
                record.update(returncode=0, path=str(artifact),
                              sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
            except Exception as error:
                record.update(returncode=1, error=repr(error))
            record["seconds"] = time.monotonic()-started
            rows.append(record)
            result = {"completed": len(rows), "total": len(ordered), "rows": rows,
                      "kernel_launches": 0, "config_library_loads": 0}
            (HERE / "cache_build.json").write_text(json.dumps(result, indent=2)+"\n")
            print(json.dumps({k: v for k, v in record.items() if k not in ("sha256", "path")}), flush=True)
        return int(any(row["returncode"] for row in rows))
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
