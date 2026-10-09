#!/usr/bin/env python3
"""Prepare two official checks and eight counter commands; never access a GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
GROUPS = ("sq_ea", "l2_tagmap", "utcl1_credits", "ta_lds")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan_path = OUT / "plans/official_characterization.json"
    plan = json.loads(plan_path.read_text())
    evidence = json.loads((OUT / "counter_evidence.json").read_text())
    helpers_spec = importlib.util.spec_from_file_location("char_inventory_helpers", OUT / "shape_inventory.py")
    helpers = importlib.util.module_from_spec(helpers_spec)
    sys.modules[helpers_spec.name] = helpers
    helpers_spec.loader.exec_module(helpers)
    registry, _ = helpers.load_cpu_registry()
    expected_targets = {
        30: {"kid": 9000, "shape": [8192, 8192, 8192], "actual_configuration": 9000,
             "BM": 256, "BN": 256, "waves_per_workgroup": 4, "workgroups": 1024,
             "waves": 4096, "dynamic_scaled_mfma": 16777216,
             "useful_f8_operations": 1099511627776, "padded_f8_operations": 1099511627776},
        31: {"kid": 9051, "shape": [1, 65536, 1536], "actual_configuration": 9051,
             "BM": 16, "BN": 32, "waves_per_workgroup": 4, "workgroups": 2048,
             "waves": 8192, "dynamic_scaled_mfma": 49152,
             "useful_f8_operations": 201326592, "padded_f8_operations": 3221225472},
    }
    assert sha(Path(plan["official_binary"])) == plan["official_binary_sha256"]
    env = {
        "AITER_AOT_IMPORT": "1", "AITER_JIT_DIR": str(OUT / "jit_baseline"),
        "GPU_ARCHS": "gfx950", "CU_NUM": "256",
        "OPUS_HIP_CLANG_PATH": "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin",
        "HIP_CLANG_PATH": "/root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build/bin",
    }
    checks, commands, files = [], [], {}
    for index, expected in expected_targets.items():
        target = plan["targets"][index]
        assert (target["kid"], target["shape"]) == (expected["kid"], expected["shape"])
        instance = registry.a8w8_mxscale_gemm_bpreshuffle_kernels_list[target["kid"]]
        assert registry.a8w8_mxscale_bpreshuffle_supports_shape(instance, *target["shape"])
        check_label = f"char_target{index}_kid{target['kid']}_check"
        check_output = OUT / "char_profiles" / check_label
        checks.append({
            "name": check_label,
            "argv": ["/opt/venv/bin/python3", str(OUT / "official_smoke.py"),
                     "--plan", str(plan_path), "--target-index", str(index),
                     "--output", str(check_output / "application.json"),
                     "--check-only", "--repetitions", "8"],
            "log": str(check_output / "check.log"),
        })
        for group in GROUPS:
            group_path = OUT / "pmc" / (group + ".txt")
            counter_names = group_path.read_text().strip().removeprefix("pmc:").split()
            assert counter_names == evidence["groups"][group]["aggregate_counters"]
            assert evidence["groups"][group]["static_budget_passed"]
            files[str(group_path)] = sha(group_path)
            label = f"char_target{index}_kid{target['kid']}_{group}"
            output_dir = OUT / "char_profiles" / label
            commands.append({
                "name": label,
                "argv": ["/opt/rocm/bin/rocprofv3", "-E", str(OUT / "gfx950_extra_counters.yaml"),
                         "-i", str(group_path), "--kernel-trace", "--kernel-include-regex",
                         "gemm_a8w8_mxfp8|opus_gemm_mxscale_bpreshuffle_reduce_kernel",
                         "-d", str(output_dir), "-o", label, "--output-format", "csv", "--",
                         "/opt/venv/bin/python3", str(OUT / "official_smoke.py"),
                         "--plan", str(plan_path), "--output", str(output_dir / "application.json"),
                         "--profile", "--target-index", str(index), "--iters", "51",
                         "--profile-rotation", "0"],
                "log": str(output_dir / "rocprof.log"),
                "parse_after_success_argv": [
                    "/opt/venv/bin/python3", str(OUT / "parse_counters.py"), str(output_dir),
                    "--output", str(output_dir / "parsed.json"), "--last", "51"],
                "required_prior_check": check_label,
                "expected_target": expected,
            })
    common = {
        "status": "prepared_cpu_only_no_gpu_execution", "env": env,
        "source_head": plan["source_head"], "plan": str(plan_path),
        "plan_sha256": sha(plan_path), "official_binary_sha256": plan["official_binary_sha256"],
        "gpu_execution_owner": "root physical-device idle lock",
    }
    check_queue = dict(common, commands=checks)
    queue = dict(common, commands=commands, group_file_sha256=files,
                 precondition="Root runs both check-only commands successfully before profiling these targets.",
                 execution_notes=[
                     "Use -i with one existing PMC group file; no shell-expanded counter list.",
                     "Root picks an idle physical GPU and checks HIP selection against PCI identity.",
                     "Each subprocess/source remains independent; retain last51 per kernel.",
                     "Do not use instrumented duration for optimization speedup decisions.",
                     "Parser omits CU/SIMD arguments because absolute GRBM-based utilization remains uncalibrated.",
                     "Target31 has M-padding: compare executed F8 operations to the padded budget, not only logical2MNK.",
                     "Group definitions/static budgets are checked; runtime acceptance for each target remains pending.",
                 ])
    (OUT / "characterization_check_queue.json").write_text(json.dumps(check_queue, indent=2) + "\n")
    (OUT / "characterization_profile_queue.json").write_text(json.dumps(queue, indent=2) + "\n")
    assert "torch" not in sys.modules
    print(json.dumps({"status": common["status"], "check_commands": len(checks),
                      "profile_commands": len(commands), "groups": list(GROUPS),
                      "output_root": str(OUT / "char_profiles")}))


if __name__ == "__main__":
    main()
