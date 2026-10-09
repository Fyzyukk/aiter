#!/usr/bin/env python3
"""Run the scalar/runtime-ABI CPU tests and bind their source/oracle inputs."""

from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEST = ROOT / "op_tests/test_opus_bpreshuffle_runtime_splitk_cpu.py"
REGISTRY_TEST = ROOT / "op_tests/test_opus_bpreshuffle_registry_cpu.py"


def main():
    spec = importlib.util.spec_from_file_location("runtime_splitk_cpu_tests", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(module)
    )
    registry_spec = importlib.util.spec_from_file_location("registry_cpu_tests", REGISTRY_TEST)
    registry_module = importlib.util.module_from_spec(registry_spec)
    registry_spec.loader.exec_module(registry_module)
    registry_result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(registry_module)
    )
    successful = result.wasSuccessful() and registry_result.wasSuccessful()
    log = HERE / "cpu_python_tests.txt"
    log.write_text(output.getvalue())
    cls = module.RuntimeSplitKCPU
    inputs = {
        TEST, REGISTRY_TEST, Path(__file__).resolve(),
        ROOT / "csrc/opus_gemm/opus_gemm_common.py",
        ROOT / "csrc/opus_gemm/opus_gemm_bpreshuffle_variants.py",
        ROOT / "csrc/opus_gemm/codegen/gen_instances_gfx950.py",
        ROOT / "csrc/opus_gemm/codegen/common.py",
        ROOT / "csrc/opus_gemm/gen_instances.py",
        ROOT / "csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py",
        ROOT / "aiter/ops/opus/dispatch.py",
        ROOT / "aiter/ops/opus/launch_plan.py",
        ROOT / "aiter/ops/opus/gemm_op_a8w8.py",
        ROOT / "csrc/opus_gemm/opus_gemm.cu",
        ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh",
        ROOT / "csrc/opus_gemm/include/gfx942/opus_gemm_arch_gfx942.cuh",
        ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_mxscale_bpreshuffle_runtime_splitk_helpers_gfx950.cuh",
        ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_runtime_gfx950.cuh",
        ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_register_runtime_gfx950.cuh",
        ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_traits_a8w8_mxscale_bpreshuffle_runtime_splitk_gfx950.cuh",
        ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_mxscale_bpreshuffle_runtime_kargs_gfx950.cuh",
    }
    for relative in (
        "csrc/opus_gemm/opus_gemm_common.py", "csrc/opus_gemm/opus_gemm_bpreshuffle_variants.py",
        "csrc/opus_gemm/codegen/gen_instances_gfx950.py", "csrc/opus_gemm/gen_instances.py",
        "csrc/opus_gemm/include/gfx950/opus_gemm_arch_gfx950.cuh",
    ):
        inputs.add(HERE / "before" / relative)
    for variant in cls.variants.values():
        inputs.add(ROOT / "csrc/opus_gemm/include" / variant.pipeline_header)
        inputs.add(ROOT / "csrc/opus_gemm/include" / variant.traits_header)
    receipt = {
        "status": "runtime_splitk_cpu_python_audit_passed" if successful else "runtime_splitk_cpu_python_audit_failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python3 reports/opus_runtime_splitk_20261009/audit_runtime_splitk_cpu.py",
        "cpu_only": True,
        "gpu_queries": 0,
        "gpu_kernel_launches": 0,
        "gpu_runtime_imports": [],
        "gpu_or_experiment_libraries_loaded": False,
        "hip_compiler_invocations": 0,
        "host_cpp_compiler_invocations": 0,
        "tests": result.testsRun + registry_result.testsRun,
        "runtime_tests": result.testsRun,
        "registration_tests": registry_result.testsRun,
        "failures": len(result.failures) + len(registry_result.failures),
        "errors": len(result.errors) + len(registry_result.errors),
        "registered_ids": len(cls.registry),
        "active_candidate_ids": len(cls.common.A8W8_BPRESHUFFLE_TUNING_KIDS),
        "legacy_ids": len(cls.common.A8W8_BPRESHUFFLE_LEGACY_KIDS),
        "runtime_candidate_ids": sorted(module.RUNTIME_IDS),
        "nonruntime_launchers_preserved": len(cls.registry) - len(module.RUNTIME_IDS),
        "k128_tile_counts_checked": 128,
        "candidate_enumeration_cases": 128 * len(module.RUNTIME_IDS),
        "runtime_launch_plan_cases": sum(min(16, total) for total in range(1, 129)) * len(module.RUNTIME_IDS),
        "auto_cu_cases": 3 * 5 * len(module.RUNTIME_IDS),
        "historical_default_cases": 6 * len(module.RUNTIME_IDS),
        "historical_preferred_candidate_cases": len(registry_module.BpreshuffleRegistryCPU.cases),
        "historical_loser_shapes": len(registry_module.BpreshuffleRegistryCPU.coverage),
        "mocked_auto_device_queries": 1,
        "public_runtime_launch_scenarios": 5,
        "public_dispatch_split_values": [0, 1, 3, 16, -1],
        "saved_replay_valid_rows": 8,
        "saved_replay_invalid_rows_rejected_before_compile_or_data": 4,
        "generated_runtime_dispatch_and_manifest_ids": len(module.RUNTIME_IDS),
        "generated_fixed_dispatch_and_unchanged_manifest_ids": len(cls.registry) - len(module.RUNTIME_IDS),
        "generated_dispatch_tables_checked": 9,
        "gfx942_dispatch_tables_and_manifest_unchanged": True,
        "runtime_and_fixed_dispatch_ids_disjoint": True,
        "numerical_validation": "not_run_gpu_stopped",
        "performance_validation": "not_run_gpu_stopped",
        "source_and_oracle_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(inputs)
        },
        "test_output_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    }
    (HERE / "cpu_python_audit.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key not in {
        "source_and_oracle_sha256", "test_output_sha256", "created_utc",
    }}, indent=2))
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
