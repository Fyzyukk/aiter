#!/usr/bin/env python3
"""Compile existing generated HIP TUs offline; never execute or load artifacts."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GENERATED = HERE.parent / "codegen"
ENV = dict(os.environ, ROCR_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
from opus_gemm_common import A8W8_BPRESHUFFLE_PIN_AGPR_KIDS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    output = HERE / "build_receipt.json"
    if output.exists():
        raise SystemExit("Refusing to overwrite a build receipt")
    reference = json.loads((ROOT / "reports/opus_flydsl_all_20261009/main_variants/build_receipt.json").read_text())
    flags = {}
    for build in reference["builds"]:
        argv = []
        for token in build["compile_argv"]:
            if token == "-c":
                break
            if not token.startswith("-I"):
                argv.append(token)
        flags[build["side"]] = argv
    metadata = json.loads((GENERATED / "all105_metadata.json").read_text())

    def compile_one(row):
        kid, name = row["kid"], row["name"]
        side = "pin24" if kid in A8W8_BPRESHUFFLE_PIN_AGPR_KIDS else "main23"
        source = GENERATED / "instances" / f"{name}_Cbf16_t.device.cu"
        obj, log = HERE / "objects" / f"kid{kid}.o", HERE / "logs" / f"kid{kid}.log"
        if obj.exists() or log.exists():
            raise RuntimeError(f"Refusing existing compile artifacts for {kid}")
        argv = flags[side] + ["-I" + str(ROOT / "csrc/include"),
                              "-I" + str(ROOT / "csrc/opus_gemm/include"),
                              "-I" + str(GENERATED), "-c", str(source), "-o", str(obj)]
        started = time.monotonic()
        proc = subprocess.run(argv, cwd=ROOT, env=ENV, capture_output=True, text=True)
        log.write_text(proc.stdout + proc.stderr)
        return dict(kid=kid, side=side, source=str(source.relative_to(ROOT)),
                    source_sha256=sha(source), impl_sha256=sha(GENERATED / "impl" / f"{name}.cuh"),
                    argv=argv, returncode=proc.returncode, seconds=time.monotonic() - started,
                    log=str(log.relative_to(ROOT)), log_sha256=sha(log),
                    object=str(obj.relative_to(ROOT)), object_sha256=sha(obj) if obj.exists() else None)

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs = {pool.submit(compile_one, row): row["kid"] for row in metadata}
        for future in concurrent.futures.as_completed(jobs):
            row = future.result()
            records.append(row)
            if row["returncode"] or len(records) % 10 == 0:
                print(json.dumps(dict(completed=len(records), total=len(metadata), kid=row["kid"],
                                      returncode=row["returncode"])), flush=True)
    records.sort(key=lambda row: row["kid"])
    result = dict(status="full105_generated_hip_object_compilation_passed" if all(row["returncode"] == 0 for row in records)
                  else "full105_generated_hip_object_compilation_failed", compile_count=len(records),
                  gpu_queries=0, library_loads=0, kernel_launches=0, libraries_linked=False,
                  compiler_sha256={side: sha(argv[0]) for side, argv in flags.items()},
                  reference_receipt_sha256=sha(ROOT / "reports/opus_flydsl_all_20261009/main_variants/build_receipt.json"),
                  metadata_sha256=sha(GENERATED / "all105_metadata.json"),
                  build_script_sha256=sha(__file__), builds=records)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], failed=[row["kid"] for row in records if row["returncode"]])), flush=True)
    return 0 if all(row["returncode"] == 0 for row in records) else 1


if __name__ == "__main__":
    raise SystemExit(run())
