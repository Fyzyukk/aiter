"""Compile generated short-K TUs on the CPU; never load or launch the output."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LLVM = Path("/root/toolchains/llvm-amdgpu-pin-op-dst-49c41889-build/bin")
SAVED = ROOT / "reports/opus_9030_wide_n_20260924/build_command.json"
saved = json.loads(SAVED.read_text())
# Retain the accepted compiler optimization flags, with an explicit target.
flags = saved[1:saved.index("-shared")]
assert "--offload-arch=gfx950" in flags and "native" not in " ".join(flags)
common = [str(LLVM / "clang++"), "-x", "hip", *flags,
          "--rocm-path=/opt/rocm", "-mllvm", "-verify-machineinstrs",
          "-I" + str(HERE / "codegen"),
          "-I" + str(ROOT / "csrc/include"),
          "-I" + str(ROOT / "csrc/opus_gemm/include"),
          "-I" + str(ROOT / "3rdparty/composable_kernel/include")]
out = HERE / "offline"
out.mkdir(exist_ok=True)
state = dict(cpu_only=True, gpu_executed=False, status="running", checks=[])
environment = dict(os.environ, GPU_ARCHS="gfx950", CU_NUM="256",
                   HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
                   ROCR_VISIBLE_DEVICES="")
start = time.time()


def run(name, args):
    before = time.time()
    with (out / (name + ".log")).open("w") as log:
        process = subprocess.run(args, cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
    state["checks"].append(dict(name=name, command=args,
                                seconds=time.time() - before,
                                returncode=process.returncode))
    if process.returncode:
        raise RuntimeError(f"{name}: see {out / (name + '.log')}")


try:
    # Import the CPU-only generator API, not its CLI native-target discovery.
    # Regenerate before compiling so these checks cannot use stale launchers.
    os.environ.update(GPU_ARCHS="gfx950", CU_NUM="256",
                      HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
                      ROCR_VISIBLE_DEVICES="")
    sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
    from gen_instances import opus_gemm_codegen
    from opus_gemm_common import a8w8_mxscale_gemm_bpreshuffle_kernels_list

    opus_gemm_codegen(str(HERE / "codegen"), istune=True).gen_instances(
        a8w8_mxscale_gemm_bpreshuffle_kernels_list
    )
    assert not any(name.split(".")[0] in ("aiter", "torch") for name in sys.modules)
    state["generated_sha256"] = {
        str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((HERE / "codegen").rglob("*")) if path.is_file()
    }
    for kid, k in ((9040, 384), (9041, 768), (9042, 1024)):
        inputs = list((HERE / "codegen/instances").glob(f"*fixedk{k}_*.device.cu"))
        assert len(inputs) == 1
        assembly = out / f"{kid}.s"
        run(str(kid), [*common, "--offload-device-only", "-S",
                       str(inputs[0]), "-o", str(assembly)])
        data = assembly.read_text()
        resources = {}
        for field in ("agpr_count", "vgpr_count", "sgpr_count",
                      "group_segment_fixed_size", "private_segment_fixed_size",
                      "vgpr_spill_count", "sgpr_spill_count", "wavefront_size",
                      "max_flat_workgroup_size"):
            values = re.findall(r"\." + field + r":\s+(\d+)", data)
            assert len(values) == 1, (kid, field, values)
            resources[field] = int(values[0])
        assert resources["private_segment_fixed_size"] == 0
        assert resources["vgpr_spill_count"] == resources["sgpr_spill_count"] == 0
        labels = {m.group(1): m.start() for m in
                  re.finditer(r"^(\.LBB\w+):", data, re.MULTILINE)}
        backward_branches = []
        for match in re.finditer(r"^\s+s_(?:branch|cbranch_\w+)\s+(\.LBB\w+)",
                                 data, re.MULTILINE):
            if labels.get(match.group(1), match.start() + 1) < match.start():
                backward_branches.append(match.group(0).strip())
        resources["backward_branches"] = backward_branches
        assert not backward_branches, (kid, backward_branches)
        resources["assembly_sha256"] = hashlib.sha256(assembly.read_bytes()).hexdigest()
        state["checks"][-1]["resources"] = resources
    host = HERE / "codegen/instances/all_instances_host_gfx950.cu"
    run("generated_host", [*common, "--offload-host-only", "-c",
                           str(host), "-o", str(out / "generated_host.o")])
    state["status"] = "passed"
except BaseException as error:
    state.update(status="failed", error=str(error))
    raise
finally:
    state["seconds"] = time.time() - start
    state["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__),
                     ROOT / "csrc/opus_gemm/opus_gemm_common.py",
                     ROOT / "csrc/opus_gemm/gen_instances.py",
                     ROOT / "csrc/opus_gemm/codegen/gen_instances_gfx950.py",
                     ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh",
                     ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_shortk_gfx950.cuh"]
    }
    (HERE / "offline_compile.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps(state, indent=2), flush=True)
