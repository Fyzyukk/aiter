#!/usr/bin/env python3
"""Build all five backend modules and the old11/current5 private kernels locally.

No old binaries or machine build manifests are consumed. --plan-only prints
commands without importing torch, compiling, or accessing a GPU. --private-only
compiles just the twelve independent libraries, with GPU visibility disabled.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from machine import CONFIG, HERE, ROOT


MODULES = ["module_aiter_core", "module_gemm_a8w8_blockscale_bpreshuffle_tune",
           "module_gemm_a8w8_blockscale_bpreshuffle_cktile_tune",
           "module_gemm_a8w8_blockscale_bpreshuffle_asm", "module_deepgemm_opus"]
REGISTERED = {9000, 9010, 9011, 9012, 9020}
JIT = HERE / "jit"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def environment(compiler):
    env = os.environ.copy()
    env.update(HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="",
               AITER_AOT_IMPORT="1", AITER_REBUILD="0", AITER_JIT_DIR=str(JIT),
               GPU_ARCHS="gfx950", CU_NUM="256", OMP_NUM_THREADS="2",
               MAX_JOBS=str(CONFIG["max_jobs"]), HIP_CLANG_PATH=str(compiler),
               OPUS_HIP_CLANG_PATH=CONFIG["opus_clang_path"], ROCM_PATH=CONFIG["rocm_path"],
               HIP_PATH=CONFIG["rocm_path"], AITER_LOG_MORE="0", AITER_SMI_MONITOR="0",
               AITER_META_DIR=str(ROOT), CK_DIR=str(ROOT / "3rdparty/composable_kernel"),
               OPUS_GEN_CO_DIR=str(ROOT / "csrc/opus_gemm/gen_co"))
    return env


def assert_source_roots(core):
    expected = dict(AITER_ROOT_DIR=ROOT, AITER_META_DIR=ROOT,
                    AITER_CSRC_DIR=ROOT / "csrc", AITER_ASM_DIR=ROOT / "hsa",
                    CK_DIR=ROOT / "3rdparty/composable_kernel")
    for key, path in expected.items():
        if Path(getattr(core, key)).resolve() != path.resolve():
            raise RuntimeError(f"AITER loaded sources outside this checkout: {key}={getattr(core, key)}")


def private_commands():
    config = json.loads((HERE / "experiments.json").read_text())
    flags = json.loads((HERE / "compile_flags.json").read_text())
    rocm = Path(CONFIG["rocm_path"])
    result = []
    for entry in config["libraries"]:
        directory = (HERE / entry["directory"]).resolve()
        command = [str(rocm / "bin/hipcc"), *flags,
                   "-I" + str(directory), "-I" + str(ROOT / "csrc/opus_gemm/include/gfx950"),
                   "-I" + str(ROOT / "csrc/opus_gemm/include"), "-I" + str(ROOT / "csrc/include")]
        if entry["name"] == "long_epilogue_sync":
            # This source's original standalone launcher disables the unused query API.
            command.append("-DOPUS_ENABLE_RUNTIME_QUERY=0")
        command += ["-shared", str(directory / "launch.hip"), "-o", str(directory / "experiments.so")]
        result.append(dict(library=entry["name"], directory=str(directory), ids=entry["ids"], command=command))
    return result


def module_worker(name):
    sys.path[:0] = [str(ROOT), str(ROOT / "csrc/opus_gemm")]
    import aiter
    from aiter.utility import dtypes
    from aiter.jit import core
    assert_source_roots(core)
    aiter.dtypes = dtypes
    if name == "module_deepgemm_opus":
        from opus_gemm_tune import _ensure_kids_compiled
        _ensure_kids_compiled(REGISTERED)
    else:
        arguments = core.get_args_of_build(name)
        arguments["md_name"] = name
        allowed = inspect.signature(core.build_module).parameters
        core.build_module(**{key: value for key, value in arguments.items() if key in allowed})
    if not (JIT / f"{name}.so").is_file():
        raise FileNotFoundError(f"Build did not produce {name}.so")


def verify_build():
    """Reject missing, modified or incomplete builds before benchmark imports torch."""
    manifest = json.loads((HERE / "build.json").read_text())
    if manifest.get("status") != "passed" or manifest.get("private_only"):
        raise ValueError("A successful complete build.py run is required before measurements")
    for path, digest in manifest["artifact_sha256"].items():
        if sha256(HERE / path) != digest:
            raise ValueError(f"Built artifact changed: {path}")
    for path, digest in manifest["source_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Source changed after build; prepare a fresh run directory: {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--private-only", action="store_true")
    parser.add_argument("--module", choices=MODULES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not CONFIG:
        parser.error("Run prepare.py first; this directory has no machine_config.json")
    if args.module:
        module_worker(args.module)
        return
    private = private_commands()
    module_commands = [dict(module=name, compiler=CONFIG["opus_clang_path"] if name == "module_deepgemm_opus"
                            else CONFIG["stock_clang_path"],
                            command=[sys.executable, "-u", str(HERE / "build.py"), "--module", name])
                       for name in ([] if args.private_only else MODULES)]
    if args.plan_only:
        print(json.dumps(dict(jit=str(JIT), private_only=args.private_only, modules=module_commands,
                              libraries=private, gpu_visibility="disabled", source="current checkout"), indent=2))
        return
    manifest_path = HERE / "build.json"
    if manifest_path.exists() or JIT.exists() or any((Path(row["directory"]) / "experiments.so").exists() for row in private):
        raise FileExistsError("Build output exists; prepare a fresh run directory instead of overwriting")
    for compiler in {CONFIG["opus_clang_path"], CONFIG["stock_clang_path"]}:
        if not (Path(compiler) / "clang++").is_file():
            raise FileNotFoundError(f"Missing clang++ in {compiler}")
    if not (Path(CONFIG["rocm_path"]) / "bin/hipcc").is_file():
        raise FileNotFoundError("Missing ROCm hipcc")
    JIT.mkdir()
    state = dict(status="running", private_only=args.private_only, gpu_executed=False,
                 start_time=time.time(), modules=[], libraries=[], compilers={},
                 head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 submodules=subprocess.check_output(["git", "submodule", "status"], cwd=ROOT, text=True).strip())
    for label, path in (("opus", CONFIG["opus_clang_path"]), ("external", CONFIG["stock_clang_path"])):
        state["compilers"][label] = subprocess.check_output([str(Path(path) / "clang++"), "--version"], text=True)
    write_json(manifest_path, state)
    try:
        for row in module_commands:
            start = time.monotonic()
            with (HERE / f"{row['module']}_build.log").open("x") as log:
                proc = subprocess.run(row["command"], cwd=ROOT, env=environment(row["compiler"]),
                                      stdout=log, stderr=subprocess.STDOUT)
            row.update(exit_code=proc.returncode, seconds=time.monotonic() - start)
            state["modules"].append(row)
            write_json(manifest_path, state)
            if proc.returncode:
                raise RuntimeError(f"Module build failed: {row['module']}; see its build log")
            print(f"Built {row['module']}", flush=True)
        for row in private:
            directory = Path(row["directory"])
            sources = {str(path): sha256(path) for path in directory.iterdir()
                       if path.suffix in {".cuh", ".hip", ".json"}}
            started = time.monotonic()
            with (directory / "build.log").open("x") as log:
                proc = subprocess.run(row["command"], cwd=ROOT, env=environment(CONFIG["opus_clang_path"]),
                                      stdout=log, stderr=subprocess.STDOUT)
            row.update(returncode=proc.returncode, elapsed_seconds=time.monotonic() - started,
                       status="passed" if proc.returncode == 0 else "failed", source_sha256=sources)
            if not proc.returncode:
                row["binary_sha256"] = sha256(directory / "experiments.so")
            write_json(directory / "build_manifest.json", row)
            state["libraries"].append(row)
            write_json(manifest_path, state)
            if proc.returncode:
                raise RuntimeError(f"Private build failed: {row['library']}; see its build.log")
            print(f"Built {row['library']} {row['ids']}", flush=True)
        if not args.private_only:
            from base import snapshot
            sources, _, _ = snapshot(JIT)
        else:
            sources = {}
        # Freeze every copied source/helper/config as well as all generated binaries.
        paths = [path for path in HERE.rglob("*") if path.is_file()
                 and "jit" not in path.relative_to(HERE).parts
                 and path.suffix in {".py", ".cuh", ".hip", ".json", ".csv"}
                 and path.name not in {"build.json", "build_manifest.json"}]
        sources.update({str(path.relative_to(ROOT)): sha256(path) for path in paths})
        artifacts = list(JIT.glob("*.so")) + [Path(row["directory"]) / "experiments.so" for row in private]
        state.update(status="passed", source_sha256=sources,
                     artifact_sha256={str(path.relative_to(HERE)): sha256(path) for path in artifacts})
    except BaseException as exc:
        state.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        state["end_time"] = time.time()
        write_json(manifest_path, state)
    print("Private libraries built; full build still required for tuning." if args.private_only
          else "JIT wrappers and private libraries rebuilt; ASM uses repository code objects. Ready for a full sweep.")


if __name__ == "__main__":
    main()
