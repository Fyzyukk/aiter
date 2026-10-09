#!/usr/bin/env python3
"""Persist the CPU registry test result and hashes of its source/oracle inputs."""

import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEST = ROOT / "op_tests/test_opus_bpreshuffle_registry_cpu.py"


def main():
    spec = importlib.util.spec_from_file_location("registration_cpu_tests", TEST)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = io.StringIO()
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(module)
    )
    (HERE / "cpu_tests.txt").write_text(output.getvalue())
    test_class = module.BpreshuffleRegistryCPU

    # Count the boundary matrix from its literal shape declarations.
    tree = ast.parse(TEST.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BpreshuffleRegistryCPU")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "test_shape_boundaries_match_frozen_contracts")
    declarations = [node for node in method.body if isinstance(node, (ast.Assign, ast.AugAssign))]
    shape_namespace = {}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), str(TEST), "exec"), shape_namespace)

    inputs = {
        TEST,
        ROOT / "csrc/opus_gemm/opus_gemm_common.py",
        ROOT / "csrc/opus_gemm/opus_gemm_bpreshuffle_variants.py",
        ROOT / "csrc/opus_gemm/opus_gemm_mxscale_bpreshuffle_tune.py",
        ROOT / "csrc/opus_gemm/codegen/gen_instances_gfx950.py",
        ROOT / "csrc/opus_gemm/gen_instances.py",
        ROOT / "aiter/jit/utils/opus_compiler.py",
        ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_mxscale_bpreshuffle_promoted_variants.json",
        ROOT / "csrc/opus_gemm/include/gfx950/opus_gemm_pipeline_a8w8_mxscale_bpreshuffle_small_lds_gfx950.cuh",
        HERE / "before/csrc/opus_gemm/opus_gemm_common.py",
        ROOT / "reports/opus_flydsl_all_20261009/candidate_catalog.json",
        ROOT / "reports/opus_flydsl_all_20261009/candidate_cases334.csv",
        ROOT / "reports/opus_flydsl_all_20261009/coverage334.csv",
        ROOT / "reports/opus_flydsl_gap_20261009/shortk9022/candidate/traits.cuh",
        ROOT / "reports/opus_flydsl_gap_20261009/shortk9022/contract.h",
        Path(__file__).resolve(),
    }
    for package in ("main_variants", "hybrid_small", "small_split", "large9030"):
        inputs.add(ROOT / "reports/opus_flydsl_all_20261009" / package / "contract.h")
    for descriptor in test_class.variants.values():
        inputs.add(ROOT / "csrc/opus_gemm/include" / descriptor.pipeline_header)
        inputs.add(ROOT / "csrc/opus_gemm/include" / descriptor.traits_header)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(inputs)}
    receipt = {
        "status": "cpu_registration_contract_audit_passed" if result.wasSuccessful() else "cpu_registration_contract_audit_failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python3 reports/opus_register92_20261009/audit_registration_cpu.py",
        "cpu_only": True,
        "gpu_queries": 0,
        "gpu_kernel_launches": 0,
        "gpu_or_experiment_libraries_loaded": False,
        "gpu_runtime_imports": [],
        "hip_compiler_invocations": 0,
        "tests": result.testsRun,
        "test_failures": len(result.failures),
        "test_errors": len(result.errors),
        "old_registry_entries_preserved": 41,
        "new_exact_variants": len(test_class.variants),
        "registry_entries": len(test_class.registry),
        "default_tuning_candidates": len(test_class.common.A8W8_BPRESHUFFLE_TUNING_KIDS),
        "legacy_entries": len(test_class.common.A8W8_BPRESHUFFLE_LEGACY_KIDS),
        "pin_compiler_entries": len(test_class.common.A8W8_BPRESHUFFLE_PIN_AGPR_KIDS),
        "generated_launchers": len(test_class.generated),
        "new_split_k_complete_calls": sum(variant.split_k > 1 for variant in test_class.variants.values()),
        "historical_preferred_candidate_cases": len(test_class.cases),
        "historical_loser_shapes": len(test_class.coverage),
        "boundary_shapes_per_variant": len(shape_namespace["shapes"]),
        "boundary_contract_comparisons": len(shape_namespace["shapes"]) * len(test_class.variants),
        "numerical_validation": "not_run_gpu_stopped",
        "performance_validation": "not_run_gpu_stopped",
        "source_and_oracle_sha256": hashes,
        "test_output_sha256": hashlib.sha256((HERE / "cpu_tests.txt").read_bytes()).hexdigest(),
    }
    (HERE / "cpu_audit.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in receipt.items() if key not in {"source_and_oracle_sha256", "created_utc", "test_output_sha256"}}, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
