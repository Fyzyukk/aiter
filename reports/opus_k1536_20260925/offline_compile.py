"""Compile K=1536 TUs on the CPU; never load or launch the output.

Adapted from reports/opus_shortk_20260925/offline_compile.py. The explicit
gfx950 target and frozen optimization flags avoid native-target discovery.
"""
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
flags = saved[1:saved.index("-shared")]
assert "--offload-arch=gfx950" in flags and "native" not in " ".join(flags)
common = [str(LLVM / "clang++"), "-x", "hip", *flags,
          "--rocm-path=/opt/rocm", "-mllvm", "-verify-machineinstrs",
          "-DOPUS_ENABLE_RUNTIME_QUERY=0",
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


class NoRuntimeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"aiter", "torch", "triton", "cupy"}:
            raise RuntimeError(f"CPU-only compilation forbids importing {fullname}")
        return None


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name, args):
    before = time.time()
    print(f"Starting {name}", flush=True)
    with (out / (name + ".log")).open("w") as log:
        process = subprocess.run(args, cwd=ROOT, env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
    record = dict(name=name, command=args, seconds=time.time() - before,
                  returncode=process.returncode)
    state["checks"].append(record)
    if process.returncode:
        raise RuntimeError(f"{name}: see {out / (name + '.log')}")
    return record


def summarize_assembly(kid, loop_unroll, assembly):
    data = assembly.read_text()
    resources = {}
    for field in ("agpr_count", "vgpr_count", "sgpr_count",
                  "group_segment_fixed_size", "private_segment_fixed_size",
                  "vgpr_spill_count", "sgpr_spill_count", "wavefront_size",
                  "max_flat_workgroup_size"):
        values = re.findall(r"\." + field + r":\s+(\d+)", data)
        assert len(values) == 1, (kid, field, values)
        resources[field] = int(values[0])
    labels = {m.group(1): m.start() for m in
              re.finditer(r"^(\.LBB\w+):", data, re.MULTILINE)}
    backward_branches = []
    for match in re.finditer(r"^\s+s_(?:branch|cbranch_\w+)\s+(\.LBB\w+)",
                             data, re.MULTILINE):
        target = labels.get(match.group(1))
        if target is not None and target < match.start():
            backward_branches.append({
                "instruction": match.group(0).strip(), "target": match.group(1),
                "target_line": data.count("\n", 0, target) + 1,
                "branch_line": data.count("\n", 0, match.start()) + 1,
            })
    counts = {}
    for mnemonic in re.findall(r"^\s+(v_mfma_\w+|s_waitcnt\w*|s_barrier)\b",
                               data, re.MULTILINE):
        counts[mnemonic] = counts.get(mnemonic, 0) + 1
    resources.update(
        backward_branches=backward_branches,
        instruction_counts_static=counts,
        mfma_count_static=sum(v for k, v in counts.items() if k.startswith("v_mfma_")),
        barrier_count_static=counts.get("s_barrier", 0),
        source_expected_dynamic_mfma_per_wave=12 * 3 * 8,
        source_expected_dynamic_barriers=2 + 11,
        loop_unroll=loop_unroll,
        assembly_sha256=sha256(assembly),
    )
    resources["has_spills"] = bool(
        resources["private_segment_fixed_size"] or
        resources["vgpr_spill_count"] or resources["sgpr_spill_count"])
    if loop_unroll == 12:
        assert not backward_branches, (kid, backward_branches)
        assert resources["mfma_count_static"] == 288, resources
    # The pair schedule intentionally permits a backward edge. Static counts
    # alone must not be interpreted as its dynamically executed instruction count.
    return resources


try:
    sys.meta_path.insert(0, NoRuntimeImports())
    os.environ.update({key: environment[key] for key in
                       ("GPU_ARCHS", "CU_NUM", "HIP_VISIBLE_DEVICES",
                        "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")})
    sys.path.insert(0, str(ROOT / "csrc/opus_gemm"))
    from gen_instances import opus_gemm_codegen
    from opus_gemm_common import a8w8_mxscale_gemm_bpreshuffle_kernels_list

    registry = a8w8_mxscale_gemm_bpreshuffle_kernels_list
    baseline = json.loads((HERE / "baseline_identity.json").read_text())
    assert set(registry) == {int(k) for k in baseline["kernels"]} | {9050, 9051}
    assert len(registry) == 14
    for kid_text, old in baseline["kernels"].items():
        instance = registry[int(kid_text)]
        assert instance.name == old["name"], (kid_text, "name")
        assert instance.k_loop_unroll is None, kid_text
        for field, expected in old["fields"].items():
            assert json.loads(json.dumps(getattr(instance, field))) == expected, (kid_text, field)
    for kid, unroll in ((9050, 12), (9051, 2)):
        assert registry[kid].fixed_k == 1536 and registry[kid].k_loop_unroll == unroll
        assert registry[kid].name.endswith(f"sfpanel12_fixedk1536_kunroll{unroll}")
    (HERE / "codegen").mkdir(exist_ok=True)
    opus_gemm_codegen(str(HERE / "codegen"), istune=True).gen_instances(registry)
    assert not any(name.split(".")[0] in {"aiter", "torch", "triton", "cupy"}
                   for name in sys.modules)
    state["generated_sha256"] = {
        str(path.relative_to(HERE)): sha256(path)
        for path in sorted((HERE / "codegen").rglob("*")) if path.is_file()
    }
    state["generated_kids"] = sorted(registry)
    state["old_instance_preservation"] = {}
    for relative, expected in baseline["generated"].items():
        if relative.startswith("impl/") or relative.endswith(".device.cu"):
            unchanged = sha256(HERE / "codegen" / relative) == expected
            state["old_instance_preservation"][relative] = unchanged
            assert unchanged, relative
    assert len(state["old_instance_preservation"]) == 24
    state["protected_source_preservation"] = {}
    for relative, expected in baseline["files"].items():
        if relative not in {"csrc/opus_gemm/opus_gemm_common.py",
                            "csrc/opus_gemm/codegen/gen_instances_gfx950.py"}:
            unchanged = sha256(ROOT / relative) == expected
            state["protected_source_preservation"][relative] = unchanged
            assert unchanged, relative
    print("Generated 14 bpreshuffle instances; all 24 old impl/device files preserved.", flush=True)
    for kid, unroll in ((9050, 12), (9051, 2)):
        inputs = list((HERE / "codegen/instances").glob(
            f"*fixedk1536_kunroll{unroll}_*.device.cu"))
        assert len(inputs) == 1, inputs
        assembly = out / f"{kid}.s"
        record = run(str(kid), [*common, "--offload-device-only", "-S",
                               str(inputs[0]), "-o", str(assembly)])
        record["resources"] = summarize_assembly(kid, unroll, assembly)
        print(json.dumps({"kid": kid, **record["resources"]}), flush=True)
    host = HERE / "codegen/instances/all_instances_host_gfx950.cu"
    run("generated_host", [*common, "--offload-host-only", "-c",
                           str(host), "-o", str(out / "generated_host.o")])
    state["status"] = "passed"
    state["resource_warning"] = any(
        check.get("resources", {}).get("has_spills", False) for check in state["checks"])
except BaseException as error:
    state.update(status="failed", error=str(error))
    raise
finally:
    state["seconds"] = time.time() - start
    state["source_sha256"] = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in [Path(__file__), SAVED,
                     ROOT / "csrc/opus_gemm/opus_gemm_common.py",
                     ROOT / "csrc/opus_gemm/gen_instances.py",
                     ROOT / "csrc/opus_gemm/codegen/gen_instances_gfx950.py",
                     ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_k1536_gfx950.cuh",
                     ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_k1536_gfx950.cuh"]
    }
    (HERE / "offline_compile.json").write_text(json.dumps(state, indent=2) + "\n")
    print(json.dumps({"status": state["status"], "seconds": state["seconds"],
                      "report": str(HERE / "offline_compile.json")}), flush=True)
