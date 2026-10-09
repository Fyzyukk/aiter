#!/usr/bin/env python3
"""Bind final registration evidence with CPU file reads only; no library loads."""

import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(relative):
    return json.loads((HERE / relative).read_text())


def write(relative, data):
    (HERE / relative).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def verify_hashes(base, hashes):
    issues = []
    for relative, expected in hashes.items():
        path = base / relative
        if not path.is_file():
            issues.append(dict(file=str(path), reason="missing"))
        elif sha(path) != expected:
            issues.append(dict(file=str(path), reason="sha256_changed"))
    return dict(files_checked=len(hashes), issues=issues, passed=not issues)


def main():
    summary = read("summary.json")
    codegen = read("codegen/codegen_receipt.json")
    cpu = read("cpu_audit.json")
    linked = read("linked_codegen_final/final_receipt.json")
    integration = read("integration_read_only_review.json")
    templates = read("read_only_template_review.json")
    catalog = read("final_catalog_doc_review.json")
    binding_checks = {
        "codegen_source_dependencies": verify_hashes(ROOT, codegen["source_sha256"]),
        "codegen_artifacts": verify_hashes(HERE / "codegen", codegen["files"]),
        "header_artifacts": verify_hashes(HERE / "header_promotion", read("header_promotion/final_manifest.json")["files"]),
        "full_link_artifacts": verify_hashes(HERE / "linked_codegen_final", linked["files"]),
        "cpu_source_and_oracle": verify_hashes(ROOT, cpu["source_and_oracle_sha256"]),
        "integration_review_inputs": verify_hashes(ROOT, integration["review_input_sha256"]),
        "template_review_inputs": verify_hashes(ROOT, templates["inputs_sha256"]),
        "catalog_export_inputs": verify_hashes(ROOT, summary["source_sha256"]),
        "catalog_outputs": verify_hashes(HERE, summary["catalog_sha256"]),
        "catalog_doc_review_inputs": verify_hashes(ROOT, catalog["inputs_sha256"]),
    }
    old_manifests = (
        "reports/opus_flydsl_all_20261009/artifact_manifest.json",
        "reports/opus_flydsl_gap_20261009/shortk9022/final_manifest.json",
    )
    prior_artifacts = {}
    for relative in old_manifests:
        path = ROOT / relative
        saved = json.loads(path.read_text())
        prior_artifacts[relative] = dict(manifest_sha256=sha(path),
            **verify_hashes(path.parent, saved["files"]))
    formal_paths = (
        "aiter/configs/model_configs/dsv4_a8w8_blockscale_bpreshuffle_opus_tuned_gemm.csv",
        "reports/opus_clang23_mixed_retune_20261008/jit/module_deepgemm_opus.so",
        "reports/opus_flydsl_comparison_20261008/upstream_mxscale_main.csv",
    )
    formal_expected = {
        **read("inputs_before.json")["files"],
        **json.loads((ROOT / old_manifests[0]).read_text())["external_inputs"],
    }
    formal = {relative: formal_expected[relative] for relative in formal_paths}
    formal_check = verify_hashes(ROOT, formal)
    assert all(check["passed"] for check in binding_checks.values()), binding_checks
    assert all(check["passed"] for check in prior_artifacts.values()), prior_artifacts
    assert formal_check["passed"], formal_check
    assert not integration["issues"]
    assert templates["status"] == "passed_source_review"
    assert catalog["status"] == "passed_final_catalog_and_document_review"
    assert not templates["canonical_primary_configurations"]["duplicate_groups"]
    assert linked["all123_host_kernel_references_defined_in_objects_and_linked_library"]
    assert linked["all64_new_scratch_and_spills_zero"]
    assert cpu["tests"] == 12 and cpu["test_errors"] == cpu["test_failures"] == 0
    assert summary["tuning_candidates"] == 92 and summary["total_registered_ids"] == 105
    diff = subprocess.run(["git", "diff", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert diff.returncode == 0, diff.stdout + diff.stderr
    child_names = (
        "cpu_audit.json", "codegen/codegen_receipt.json", "header_promotion/final_receipt.json",
        "header_promotion/template_groups92.json", "linked_codegen_final/final_receipt.json",
        "integration_read_only_review.json", "read_only_template_review.json", "final_catalog_doc_review.json",
    )
    result = dict(
        status="registration_and_template_consolidation_complete_gpu_validation_pending",
        public_tuning_candidates=92, registered_ids=105, old_entries_preserved=41,
        new_exact_variants=64, shared_primary_gemm_templates=18, shared_reducer_templates=1,
        completely_duplicate_public_configurations=0,
        full_hip_objects_compiled=105, full_link_passed=True,
        host_launch_stub_references_resolved=123,
        new64_scratch_vgpr_spills_sgpr_spills_zero=True,
        cpu_tests=dict(
            command="python -m pytest -q op_tests/test_opus_bpreshuffle_registry_cpu.py op_tests/test_opus_mixed_compiler.py",
            tests_passed=18, failures=0, result="18 passed in 0.43s",
            execution_evidence="Root tool result earlier in this same registration task; combined console output was not saved to a file.",
            registered_contract_tests_saved=12,
            saved_registration_receipt="cpu_audit.json", saved_registration_log="cpu_tests.txt",
            boundary_contract_comparisons=36864,
        ),
        current_source_and_artifact_hash_checks=binding_checks,
        frozen_prior_report_files=prior_artifacts,
        formal_artifacts_unchanged=dict(**formal_check, sha256=formal),
        old_external_hash_note="Only frozen report files are checked against old manifests. Their external current-source hashes describe the pre-registration snapshots and are intentionally historical.",
        catalog_doc_review_status=catalog["status"],
        child_receipt_sha256={relative: sha(HERE / relative) for relative in child_names},
        git_diff_check=dict(returncode=diff.returncode, passed=True),
        gpu_queries=0, gpu_runtime_imports=0, generated_libraries_loaded=0,
        gpu_kernel_launches=0, benchmark_runs=0, automatic_gpu_wait_queues=0,
        numerical_validation="not_run_gpu_stopped", performance_validation="not_run_gpu_stopped",
        formal_tuning_updated=False,
        finalize_script_sha256=sha(__file__),
    )
    write("verification.json", result)
    sources = dict(codegen["source_sha256"])
    for relative in (
        "aiter/jit/utils/opus_compiler.py", "op_tests/test_opus_bpreshuffle_registry_cpu.py",
        "op_tests/test_opus_mixed_compiler.py", "csrc/opus_gemm/README.md", "HANDOFF_MXFP8.md",
    ):
        sources[relative] = sha(ROOT / relative)
    files = {
        str(path.relative_to(HERE)): sha(path)
        for path in sorted(HERE.rglob("*"))
        if path.is_file() and path.name != "artifact_manifest.json"
        and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    write("artifact_manifest.json", dict(
        status=result["status"], scope="All new report artifacts, including superseded inspection attempts; no generated library was loaded.",
        files=files, source_dependencies=sources, formal_inputs=formal,
        frozen_prior_manifest_sha256={name: check["manifest_sha256"] for name, check in prior_artifacts.items()},
        gpu_queries=0, gpu_runtime_imports=0, generated_libraries_loaded=0, gpu_kernel_launches=0,
    ))
    print(json.dumps(dict(status=result["status"], report_files_bound=len(files),
        source_dependencies_bound=len(sources),
        frozen_prior_files_checked=sum(check["files_checked"] for check in prior_artifacts.values())), indent=2))


if __name__ == "__main__":
    main()
