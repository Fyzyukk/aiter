#!/usr/bin/env python3
"""Fresh device compilation of extra tuples; no HIP imports or library loads."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
from gen_instances import opus_gemm_codegen
from csrc.opus_gemm.opus_gemm_bpreshuffle_config import construct_config, kernel_instance_from_config


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    out = HERE / "extra_device_build"
    out.mkdir(exist_ok=False)
    extras = json.loads((HERE / "extra_configs.json").read_text())
    receipt = json.loads((ROOT / "reports/opus_flydsl_all_20261009/main_variants/build_receipt.json").read_text())
    flags = {}
    for row in receipt["builds"]:
        argv = row["compile_argv"][:row["compile_argv"].index("-c")]
        flags[row["side"]] = [v for v in argv if not v.startswith("-I")]
    env = {**os.environ, "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": ""}
    jobs = []
    for row in extras:
        c = construct_config(row["pipeline"], config=row["compile_params"])
        directory = out / row["identity"]
        directory.mkdir()
        opus_gemm_codegen(str(directory)).gen_instances({1: kernel_instance_from_config(c)})
        source = next((directory / "instances").glob("*.device.cu"))
        obj = directory / "device.o"
        argv = flags["pin24" if c.pin_agpr else "main23"] + [
            "-D__HIPCC_RTC__", "-mllvm", "--disable-machine-licm",
            "-DENABLE_CK=0", "-I"+str(ROOT / "csrc/include"),
            "-I"+str(ROOT / "csrc/opus_gemm/include"), "-I"+str(directory),
            "-c", str(source), "-o", str(obj)]
        jobs.append((row, directory, source, obj, argv))

    def build(job):
        row, directory, source, obj, argv = job
        started = time.monotonic()
        p = subprocess.run(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (directory / "device.log").write_text(p.stdout)
        result = {**row, "returncode": p.returncode, "seconds": time.monotonic()-started,
                  "argv": argv, "source_sha256": sha(source), "object": str(obj),
                  "object_sha256": sha(obj) if obj.exists() else None,
                  "log": str(directory / "device.log")}
        (directory / "receipt.json").write_text(json.dumps(result, indent=2)+"\n")
        return result

    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [executor.submit(build, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            (out / "receipt.json").write_text(json.dumps({"plan_sha256": sha(HERE / "plan.json"),
                "completed": len(rows), "total": len(jobs), "gpu_operations": 0,
                "rows": rows}, indent=2)+"\n")
            print(json.dumps({"completed": len(rows), "total": len(jobs), "pipeline": row["pipeline"],
                              "identity": row["identity"], "returncode": row["returncode"],
                              "seconds": row["seconds"]}), flush=True)
    return int(any(row["returncode"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
